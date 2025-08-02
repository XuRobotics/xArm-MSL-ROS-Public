"""
MIT License

Copyright (c) 2025 Xu Liu, Chengyang He, Gadiel Sznaier Camps at 
Multi-Robot Systems Lab, Stanford University (with Github Copilot)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

############################################################################################################################
# The data structure after this processing script is as follows:
# data.zarr/
# │
# ├── data/
# │   ├── timestamp                 # (N,) array: timestamps of synchronized samples
# │   ├── arm1_action               # (N, 10): [x, y, z, rot6D (6 values), gripper opening] # gripper opening is fixed to 0
# │   ├── arm1_eef_quat             # (N, 4): quaternion orientation [x, y, z, w]
# │   ├── arm1_robot_eef_pos        # (N, 3): end-effector position (x, y, z)
# │   ├── arm2_action               # (N, 10), same structure as arm1_action
# │   ├── arm2_eef_quat             # (N, 4), quaternion (x, y, z, w)
# │   └── arm2_robot_eef_pos        # (N, 3), end-effector position (x, y, z)
# │
# └── meta/
#     └── episode_ends              # indices marking the end of each episode/bag
# 
# ├── 00/
# │   ├── 1.mp4                      (wrist camera arm1 video)
# │   ├── 3.mp4                      (fixed camera video)
# │   └── 4.mp4                      (wrist camera arm2 video)
# 
# ├── 01/
# │   ├── 1.mp4
# │   ├── 3.mp4
# │   └── 4.mp4
#
# ...
############################################################################################################################


# BAG DATA EXAMPLE:
# path:        xarm_demo_2025-03-08-15-32-32.bag
# version:     2.0
# duration:    42.7s
# start:       Mar 08 2025 15:32:32.83 (1741476752.83)
# end:         Mar 08 2025 15:33:15.49 (1741476795.49)
# size:        2.5 GB
# messages:    26239
# compression: none [1225/1225 chunks]
# types:       geometry_msgs/PoseStamped [d3812c3cbc69362b77dc0b19b345f8f5]
#              sensor_msgs/CameraInfo    [c9a58c1b0b154e0e6da7578cb991d214]
#              sensor_msgs/Image         [060021388200f6f0f447d0fcd9c64743]
#              std_msgs/Float32          [73fcbf46b49191e672908e50842a83d4]
#              tf2_msgs/TFMessage        [94810edda583a504dfda3829e70d7eec]
# topics:      /fixed_camera/color/camera_info       640 msgs    : sensor_msgs/CameraInfo   
#              /fixed_camera/color/image_raw         640 msgs    : sensor_msgs/Image        
#              /robot_end_effector_opening          1760 msgs    : std_msgs/Float32         
#              /robot_end_effector_opening_arm2     1759 msgs    : std_msgs/Float32         
#              /robot_end_effector_pose             1760 msgs    : geometry_msgs/PoseStamped
#              /robot_end_effector_pose_arm2        1759 msgs    : geometry_msgs/PoseStamped
#              /tf                                 15362 msgs    : tf2_msgs/TFMessage       
#              /tf_static                              3 msgs    : tf2_msgs/TFMessage        (3 connections)
#              /wrist_camera/color/camera_info       639 msgs    : sensor_msgs/CameraInfo   
#              /wrist_camera/color/image_raw         639 msgs    : sensor_msgs/Image        
#              /wrist_camera_2/color/camera_info     639 msgs    : sensor_msgs/CameraInfo   
#              /wrist_camera_2/color/image_raw       639 msgs    : sensor_msgs/Image



import rosbag 
import os
import numpy as np
import zarr
import cv2
from cv_bridge import CvBridge
import imageio
import argparse
import time
from scipy.spatial.transform import Rotation as R

# Install necessary packages
try:
    import zarr
    import imageio
except ImportError:
    print("Installing required packages...")
    os.system("pip install zarr imageio[ffmpeg]")

def ensure_dir(directory, check=False):
    if not check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created for data store.")
    # if already exists, delete it with user's key in confirmation (double confirm)
    elif os.path.exists(directory):
        response = input(f"Directory '{directory}' already exists. Delete it? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -r {directory}")
        else:
            print("Exiting...")
            exit()
        os.makedirs(directory)
    else:
        # confirm to create the directory
        response = input(f"Directory '{directory}' does not exist. Create it? (y/n): ")
        if response.lower() == 'y':
            os.makedirs(directory)
            print(f"Directory '{directory}' created for data store.")

def get_closest_message(target_sec, messages, max_diff=0.1):
    """
    Given a target time (in seconds) and a list of messages (each a tuple (topic, msg, t)),
    return the message with the closest timestamp if the difference is less than max_diff.
    Otherwise, return None.
    """
    best_msg = None
    best_diff = float('inf')
    for entry in messages:
        msg_time = entry[2].to_sec()
        diff = abs(msg_time - target_sec)
        if diff < best_diff:
            best_diff = diff
            best_msg = entry
    if best_diff > max_diff:
        return None
    return best_msg

def synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, pose_arm2_msgs, wrist_arm2_msgs, target_rate):
    """
    Synchronize pose and image messages by sampling at a period defined by target_rate.
    For each sample time, the function finds the closest pose message, fixed-camera image,
    and wrist-camera image.
    """
    T = 1.0 / target_rate
    if not pose_msgs or not fixed_msgs or not wrist_msgs or not pose_arm2_msgs or not wrist_arm2_msgs:
        # ask user to press enter to continue, check if the bag is missing some topics
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("No messages found for one or more topics. Skipping this bag.")
        print("Check if the bag is missing some topics.")
        print("But you can continue to process the next bag...")
        input("Press Enter to continue...")
        return [], []
    
    # Ensure the lists are sorted by time
    pose_msgs = sorted(pose_msgs, key=lambda x: x[2].to_sec())
    fixed_msgs = sorted(fixed_msgs, key=lambda x: x[2].to_sec())
    wrist_msgs = sorted(wrist_msgs, key=lambda x: x[2].to_sec())
    # gripper_opening_msgs = sorted(gripper_opening_msgs, key=lambda x: x[2].to_sec())
    pose_arm2_msgs = sorted(pose_arm2_msgs, key=lambda x: x[2].to_sec())
    wrist_arm2_msgs = sorted(wrist_arm2_msgs, key=lambda x: x[2].to_sec())
    
    
    # Define the overlapping time window
    start_time = max(pose_msgs[0][2].to_sec(), fixed_msgs[0][2].to_sec(), wrist_msgs[0][2].to_sec(), pose_arm2_msgs[0][2].to_sec(), wrist_arm2_msgs[0][2].to_sec())
    end_time = min(pose_msgs[-1][2].to_sec(), fixed_msgs[-1][2].to_sec(), wrist_msgs[-1][2].to_sec(), pose_arm2_msgs[-1][2].to_sec(), wrist_arm2_msgs[-1][2].to_sec())
    
    synced_data = []
    synced_data_no_images = []
    t_sample = start_time
    pose_entry, fixed_entry, wrist_entry, pose_arm2_entry, wrist_arm2_entry = None, None, None, None, None
    while t_sample <= end_time:
        pose_entry = get_closest_message(t_sample, pose_msgs)
        fixed_entry = get_closest_message(t_sample, fixed_msgs)
        wrist_entry = get_closest_message(t_sample, wrist_msgs)
        pose_arm2_entry = get_closest_message(t_sample, pose_arm2_msgs)
        wrist_arm2_entry = get_closest_message(t_sample, wrist_arm2_msgs)
        
        if pose_entry is None or fixed_entry is None or wrist_entry is None or pose_arm2_entry is None or wrist_arm2_entry is None:
            t_sample += T
            print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Skipping sample at time {t_sample} due to unable to sync messages.")
            continue
        synced_data.append({
            "timestamp": t_sample,
            "pose_arm1": pose_entry,
            "fixed": fixed_entry,
            "wrist": wrist_entry, 
            "pose_arm2": pose_arm2_entry,
            "wrist_arm2": wrist_arm2_entry
        })
        synced_data_no_images.append({
            "timestamp": t_sample,
            "pose_arm1": pose_entry,
            "pose_arm2": pose_arm2_entry,
        })        
        t_sample += T
    return synced_data, synced_data_no_images

def save_synchronized_video(synced_data, output_dir, bag_index, bridge, target_rate):
    """
    For each synchronized sample, convert the ROS image messages into CV images and
    write them into two video files (one per camera) at the target_rate.
    """
    print(f"Processing synchronized video for bag {bag_index}...")
    camera_dir = os.path.join(output_dir, f"{bag_index:02d}") 
    ensure_dir(camera_dir)
    
    fixed_camera_path = os.path.join(camera_dir, "3.mp4")
    wrist_camera_path = os.path.join(camera_dir, "1.mp4")
    wrist_arm2_camera_path = os.path.join(camera_dir, "4.mp4")
    
    

    fixed_writer = imageio.get_writer(fixed_camera_path, fps=target_rate, codec='libx264', format='FFMPEG')
    wrist_writer = imageio.get_writer(wrist_camera_path, fps=target_rate, codec='libx264', format='FFMPEG')
    wrist_arm2_writer = imageio.get_writer(wrist_arm2_camera_path, fps=target_rate, codec='libx264', format='FFMPEG')
    
    for entry in synced_data:
        try:
            fixed_cv_image = bridge.imgmsg_to_cv2(entry["fixed"][1], "bgr8")
            wrist_cv_image = bridge.imgmsg_to_cv2(entry["wrist"][1], "bgr8")
            wrist_arm2_cv_image = bridge.imgmsg_to_cv2(entry["wrist_arm2"][1], "bgr8")
            fixed_rgb = cv2.cvtColor(fixed_cv_image, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_cv_image, cv2.COLOR_BGR2RGB)
            wrist_arm2_rgb = cv2.cvtColor(wrist_arm2_cv_image, cv2.COLOR_BGR2RGB)
            fixed_writer.append_data(fixed_rgb)
            wrist_writer.append_data(wrist_rgb)
            wrist_arm2_writer.append_data(wrist_arm2_rgb)
        except Exception as e:
            print(f"Error processing image for bag {bag_index}: {e}")
    
    fixed_writer.close()
    wrist_writer.close()
    wrist_arm2_writer.close()
    print(f"Finished processing synchronized video for bag {bag_index}.")



def quat_to_six_d(quat):
    """
    Converts a quaternion to a 6D rotation representation.
    
    Args:
        quat (array-like): A quaternion [x, y, z, w]
        
    Returns:
        np.ndarray: A 6D rotation representation (first two columns of the rotation matrix)
    """
    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat(quat).as_matrix()
    
    # Extract the first two columns
    rot_6d = rot_matrix[:, :2].flatten()
    
    return rot_6d

def save_data(synced_pose_data, episode_ends, output_dir, chunk_size=(170,)):
    """
    Save synchronized robot pose data (and computed 'action' from the next sample)
    into a Zarr store.
    """
    print("Saving data and metadata to Zarr format...")
    root_store = zarr.open(os.path.join(output_dir, "data.zarr"), mode='w')
    data_store = root_store.create_group("data")
    meta_store = root_store.create_group("meta")
    
    
        

    # data structure:
    # - timestamp (~, )
    # - arm1_action (~, 10) -- x, y, z, 6D orientation, gripper position # gripper opening is default to 0 since we are always grabbing the "pusher" of the blocks
    # - arm2_action (~, 10) -- x, y, z, 6D orientation, gripper position # gripper opening is default to 0 since we are always grabbing the "pusher" of the blocks
    # - arm1_eef_quat (~, 4) -- x, y, z, w
    # - arm2_eef_quat (~, 4) -- x, y, z, w
    # - arm1_robot_eef_pos (~, 3) -- x, y, z
    # - arm2_robot_eef_pos (~, 3) -- x, y, z
    data_store.create_dataset("timestamp", shape=(0,), chunks=chunk_size, dtype=np.float64)
    data_store.create_dataset("arm1_action", shape=(0, 10), chunks=(chunk_size[0], 10), dtype=np.float64)
    data_store.create_dataset("arm2_action", shape=(0, 10), chunks=(chunk_size[0], 10), dtype=np.float64)
    # data_store.create_dataset("action", shape=(0, 20), chunks=(chunk_size[0], 20), dtype=np.float64)
    data_store.create_dataset("arm1_eef_quat", shape=(0, 4), chunks=(chunk_size[0], 4), dtype=np.float64)
    data_store.create_dataset("arm2_eef_quat", shape=(0, 4), chunks=(chunk_size[0], 4), dtype=np.float64)
    data_store.create_dataset("arm1_robot_eef_pos", shape=(0, 3), chunks=(chunk_size[0], 3), dtype=np.float64)
    data_store.create_dataset("arm2_robot_eef_pos", shape=(0, 3), chunks=(chunk_size[0], 3), dtype=np.float64)

    
    # timestamps, position, orientation, actions, position_arm2, orientation_arm2, actions_arm2 = [], [], [], [], [], [], []
    timestamps, position, orientation, action_two_arms, position_arm2, orientation_arm2 = [], [], [], [], [], []
    actions = []
    actions_arm2 = []
    
    # Use synchronized pose data. Here, "arm1_action" is taken as the pose from the next time step.
    for i in range(len(synced_pose_data) - 1):  # Skip last sample for action
        _, msg, t = synced_pose_data[i]["pose_arm1"]
        timestamps.append(t.to_sec())
        position.append([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])        
        # Convert quaternion to 6D orientation representation
        quat = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]        
        orientation.append(quat)
        
        # add for arm2
        _, msg_arm2, t_arm2 = synced_pose_data[i]["pose_arm2"]
        position_arm2.append([
            msg_arm2.pose.position.x,
            msg_arm2.pose.position.y,
            msg_arm2.pose.position.z,
        ])
        # Convert quaternion to 6D orientation representation
        quat_arm2 = [
            msg_arm2.pose.orientation.x,
            msg_arm2.pose.orientation.y,
            msg_arm2.pose.orientation.z,
            msg_arm2.pose.orientation.w
        ]
        orientation_arm2.append(quat_arm2)
        
        next_msg = synced_pose_data[i+1]["pose_arm1"][1]
        next_msg_quat = [
            next_msg.pose.orientation.x,
            next_msg.pose.orientation.y,
            next_msg.pose.orientation.z,
            next_msg.pose.orientation.w
        ]
        orientation_6d = quat_to_six_d(next_msg_quat)
        
        next_msg_arm2 = synced_pose_data[i+1]["pose_arm2"][1]
        next_msg_quat_arm2 = [
            next_msg_arm2.pose.orientation.x,
            next_msg_arm2.pose.orientation.y,
            next_msg_arm2.pose.orientation.z,
            next_msg_arm2.pose.orientation.w
        ]
        orientation_6d_arm2 = quat_to_six_d(next_msg_quat_arm2)
        
        # action's last dim gripper opening = 1 if gripper_msg.data > 850 / 2, else 0            
        # action is 10-dim vector: x, y, z, 6D orientation (orientation_6d), gripper opening
        actions.append([
            next_msg.pose.position.x,
            next_msg.pose.position.y,
            next_msg.pose.position.z,
            orientation_6d[0],
            orientation_6d[1],
            orientation_6d[2],
            orientation_6d[3],
            orientation_6d[4],
            orientation_6d[5],
            0 
        ])
        
        actions_arm2.append([
            next_msg_arm2.pose.position.x,
            next_msg_arm2.pose.position.y,
            next_msg_arm2.pose.position.z,
            orientation_6d_arm2[0],
            orientation_6d_arm2[1],
            orientation_6d_arm2[2],
            orientation_6d_arm2[3],
            orientation_6d_arm2[4],
            orientation_6d_arm2[5],
            0
        ])
        # action_two_arms.append([
        #     next_msg.pose.position.x,
        #     next_msg.pose.position.y,
        #     next_msg.pose.position.z,
        #     orientation_6d[0],
        #     orientation_6d[1],
        #     orientation_6d[2],
        #     orientation_6d[3],
        #     orientation_6d[4],
        #     orientation_6d[5],
        #     0,
        #     next_msg_arm2.pose.position.x,
        #     next_msg_arm2.pose.position.y,
        #     next_msg_arm2.pose.position.z,
        #     orientation_6d_arm2[0],
        #     orientation_6d_arm2[1],
        #     orientation_6d_arm2[2],
        #     orientation_6d_arm2[3],
        #     orientation_6d_arm2[4],
        #     orientation_6d_arm2[5],
        #     0
        # ])
            
            
    
    data_store["timestamp"].append(np.array(timestamps))
    data_store["arm1_robot_eef_pos"].append(np.array(position))
    data_store["arm1_action"].append(np.array(actions))
    data_store["arm1_eef_quat"].append(np.array(orientation))
    data_store["arm2_robot_eef_pos"].append(np.array(position_arm2))
    data_store["arm2_action"].append(np.array(actions_arm2))
    # data_store["action"].append(np.array(action_two_arms))    
    data_store["arm2_eef_quat"].append(np.array(orientation_arm2))
    
    meta_store.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64), chunks=(len(episode_ends),))
    # print out episode ends and details on the data info
    print(f"Episode ends: {episode_ends}")
    print(f"Total samples: {len(timestamps)}")
    # total actions should be one less than total samples
    # print(f"Total actions: {len(action_two_arms)}")
    # timestamps should be equal to total samples
    print(f"Total timestamps: {len(timestamps)}")
    print("Data and metadata saved successfully.")

def process_rosbags(bags_folder, output_dir, target_rate):
    """
    For each ROS bag in the given folder, read and separate the robot pose and image messages,
    synchronize them based on the target_rate, write synchronized videos, and accumulate the
    synchronized pose data for saving into Zarr.
    """
    ensure_dir(output_dir, check=True)
    bag_files = [os.path.join(bags_folder, f) for f in os.listdir(bags_folder) if f.endswith(".bag")]
    
    global_synced_pose_data = []
    episode_ends = []
    total_samples = 0
    bridge = CvBridge()
    
    
    bag_index = 0
    # for bag_index, bag_file in enumerate(bag_files):
    for bag_file in bag_files:
        print(f"Processing bag {bag_index}: {bag_file}")
        bag = rosbag.Bag(bag_file, "r")
        
        pose_msgs = []
        pose_arm2_msgs = []
        fixed_msgs = []
        wrist_msgs = []
        wrist_arm2_msgs = []
        
        for topic, msg, t in bag.read_messages():
            if topic == '/robot_end_effector_pose':
                pose_msgs.append((topic, msg, t))
            elif topic == '/fixed_camera/color/image_raw':
                fixed_msgs.append((topic, msg, t))
            elif topic == '/wrist_camera/color/image_raw':
                wrist_msgs.append((topic, msg, t))
            elif topic == '/robot_end_effector_pose_arm2':
                pose_arm2_msgs.append((topic, msg, t))
            elif topic == '/wrist_camera_2/color/image_raw':
                wrist_arm2_msgs.append((topic, msg, t))
        bag.close()
        
        # Synchronize messages at the given target_rate.
        synced_data, synced_data_no_images = synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, pose_arm2_msgs, wrist_arm2_msgs, target_rate)
        if not synced_data:
            print(f"No synchronized data for bag {bag_index}")
            continue
        
        # write synchronized video for each bag 
        save_synchronized_video(synced_data, output_dir, bag_index, bridge, target_rate)
        
        # Append the synchronized pose data (and corresponding timestamps) to the global list.
        global_synced_pose_data.extend(synced_data_no_images)
        total_samples += len(synced_data_no_images)
        episode_ends.append(total_samples - 1)  # Mark end index for this bag
        
        print(f"Bag {bag_index} synchronized samples: {len(synced_data_no_images)}")
    
        bag_index += 1
        
    save_data(global_synced_pose_data, episode_ends, output_dir)
    print("ROS bag processing complete. Data saved in:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_folder', type=str, default='/home/sam/bags/msl_bags/decentralized_policy_4_23',
                        help='Folder containing ROS bag files.')
    parser.add_argument('--output_dir', type=str, default='/home/sam/bags/msl_bags/rosbag_processed_data',
                        help='Output directory to save synchronized data.')
    parser.add_argument('--target_rate', type=float, default=10.0,
                        help='Target synchronization rate in Hz (i.e., reduced frequency).')
    args = parser.parse_args()
    
    process_rosbags(args.bags_folder, args.output_dir, args.target_rate)