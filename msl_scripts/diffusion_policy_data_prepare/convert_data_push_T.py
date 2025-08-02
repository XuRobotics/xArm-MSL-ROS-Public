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


import rosbag
import os
import numpy as np
import zarr
import cv2
from cv_bridge import CvBridge
import imageio
import argparse

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


def get_closest_message(target_sec, messages, max_diff=1/30.0):
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

def synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, target_rate):
    """
    Synchronize pose and image messages by sampling at a period defined by target_rate.
    For each sample time, the function finds the closest pose message, fixed-camera image,
    and wrist-camera image.
    """
    T = 1.0 / target_rate
    if not pose_msgs or not fixed_msgs or not wrist_msgs:
        return []
    
    # Ensure the lists are sorted by time
    pose_msgs = sorted(pose_msgs, key=lambda x: x[2].to_sec())
    fixed_msgs = sorted(fixed_msgs, key=lambda x: x[2].to_sec())
    wrist_msgs = sorted(wrist_msgs, key=lambda x: x[2].to_sec())
    
    # Define the overlapping time window
    start_time = max(pose_msgs[0][2].to_sec(), fixed_msgs[0][2].to_sec(), wrist_msgs[0][2].to_sec())
    end_time = min(pose_msgs[-1][2].to_sec(), fixed_msgs[-1][2].to_sec(), wrist_msgs[-1][2].to_sec())
    
    synced_data = []
    synced_data_no_images = []
    t_sample = start_time
    while t_sample <= end_time:
        pose_entry = get_closest_message(t_sample, pose_msgs)
        fixed_entry = get_closest_message(t_sample, fixed_msgs)
        wrist_entry = get_closest_message(t_sample, wrist_msgs)
        if pose_entry is None or fixed_entry is None or wrist_entry is None:
            t_sample += T
            print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Skipping sample at time {t_sample} due to unable to sync messages.")
            continue
        synced_data.append({
            "timestamp": t_sample,
            "pose": pose_entry,
            "fixed": fixed_entry,
            "wrist": wrist_entry
        })
        synced_data_no_images.append({
            "timestamp": t_sample,
            "pose": pose_entry
        })
        t_sample += T
    return synced_data, synced_data_no_images

def save_synchronized_video(synced_data, output_dir, bag_index, bridge, target_rate):
    """
    For each synchronized sample, convert the ROS image messages into CV images and
    write them into two video files (one per camera) at the target_rate.
    """
    print(f"Processing synchronized video for bag {bag_index}...")
    fixed_camera_dir = os.path.join(output_dir, f"{bag_index:02d}", "fixed_camera")
    wrist_camera_dir = os.path.join(output_dir, f"{bag_index:02d}", "wrist_camera")
    ensure_dir(fixed_camera_dir)
    ensure_dir(wrist_camera_dir)
    
    fixed_camera_path = os.path.join(fixed_camera_dir, "video.mp4")
    wrist_camera_path = os.path.join(wrist_camera_dir, "video.mp4")
    
    fixed_writer = imageio.get_writer(fixed_camera_path, fps=target_rate, codec='libx264', format='FFMPEG')
    wrist_writer = imageio.get_writer(wrist_camera_path, fps=target_rate, codec='libx264', format='FFMPEG')
    
    for entry in synced_data:
        try:
            fixed_cv_image = bridge.imgmsg_to_cv2(entry["fixed"][1], "bgr8")
            wrist_cv_image = bridge.imgmsg_to_cv2(entry["wrist"][1], "bgr8")
            fixed_rgb = cv2.cvtColor(fixed_cv_image, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_cv_image, cv2.COLOR_BGR2RGB)
            fixed_writer.append_data(fixed_rgb)
            wrist_writer.append_data(wrist_rgb)
        except Exception as e:
            print(f"Error processing image for bag {bag_index}: {e}")
    
    fixed_writer.close()
    wrist_writer.close()
    print(f"Finished processing synchronized video for bag {bag_index}.")

def save_data(synced_pose_data, episode_ends, output_dir, chunk_size=(170,)):
    """
    Save synchronized robot pose data (and computed 'action' from the next sample)
    into a Zarr store.
    """
    print("Saving data and metadata to Zarr format...")
    root_store = zarr.open(os.path.join(output_dir, "data.zarr"), mode='w')
    data_store = root_store.create_group("data")
    meta_store = root_store.create_group("meta")
    
    data_store.create_dataset("timestamp", shape=(0,), chunks=chunk_size, dtype=np.float64)
    data_store.create_dataset("robot_eef_pose", shape=(0, 6), chunks=(chunk_size[0], 6), dtype=np.float64)
    data_store.create_dataset("action", shape=(0, 6), chunks=(chunk_size[0], 6), dtype=np.float64)
    data_store.create_dataset("robot_eef_pose_vel", shape=(0, 6), chunks=(chunk_size[0], 6), dtype=np.float64)
    data_store.create_dataset("robot_joint", shape=(0, 6), chunks=(chunk_size[0], 6), dtype=np.float64)
    data_store.create_dataset("robot_joint_vel", shape=(0, 6), chunks=(chunk_size[0], 6), dtype=np.float64)
    data_store.create_dataset("stage", shape=(0,), chunks=chunk_size, dtype=np.int64)
    
    timestamps, poses, pose_vels, joints, joint_vels, stages, actions = [], [], [], [], [], [], []
    
    # Use synchronized pose data. Here, "action" is taken as the pose from the next time step.
    for i in range(len(synced_pose_data) - 1):  # Skip last sample for action
        _, msg, t = synced_pose_data[i]["pose"]
        timestamps.append(t.to_sec())
        poses.append([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            0, 0, 0  # Placeholder for orientation
        ])
        pose_vels.append([0] * 6)  # Placeholder for velocity
        joints.append([0] * 6)     # Placeholder for joint data
        joint_vels.append([0] * 6) # Placeholder for joint velocity
        stages.append(0)           # Placeholder for stage
        
        next_msg = synced_pose_data[i+1]["pose"][1]
        actions.append([
            next_msg.pose.position.x,
            next_msg.pose.position.y,
            next_msg.pose.position.z,
            0, 0, 0  # Placeholder for orientation
        ])
    
    data_store["timestamp"].append(np.array(timestamps))
    data_store["robot_eef_pose"].append(np.array(poses))
    data_store["action"].append(np.array(actions))
    data_store["robot_eef_pose_vel"].append(np.array(pose_vels))
    data_store["robot_joint"].append(np.array(joints))
    data_store["robot_joint_vel"].append(np.array(joint_vels))
    data_store["stage"].append(np.array(stages, dtype=np.int64))
    
    meta_store.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64), chunks=(len(episode_ends),))
    # print out episode ends and details on the data info
    print(f"Episode ends: {episode_ends}")
    print(f"Total samples: {len(timestamps)}")
    # total actions should be one less than total samples
    print(f"Total actions: {len(actions)}")
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
    
    for bag_index, bag_file in enumerate(bag_files):
        print(f"Processing bag {bag_index}: {bag_file}")
        bag = rosbag.Bag(bag_file, "r")
        
        pose_msgs = []
        fixed_msgs = []
        wrist_msgs = []
        
        for topic, msg, t in bag.read_messages():
            if topic == '/robot_end_effector_pose':
                pose_msgs.append((topic, msg, t))
            elif topic == '/fixed_camera/color/image_raw':
                fixed_msgs.append((topic, msg, t))
            elif topic == '/wrist_camera/color/image_raw':
                wrist_msgs.append((topic, msg, t))
        
        bag.close()
        
        # Synchronize messages at the given target_rate.
        synced_data, synced_data_no_images = synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, target_rate)
        if not synced_data:
            print(f"No synchronized data for bag {bag_index}")
            continue
        
        save_synchronized_video(synced_data, output_dir, bag_index, bridge, target_rate)
        
        # Append the synchronized pose data (and corresponding timestamps) to the global list.
        global_synced_pose_data.extend(synced_data_no_images)
        total_samples += len(synced_data_no_images)
        episode_ends.append(total_samples - 1)  # Mark end index for this bag
        
        print(f"Bag {bag_index} synchronized samples: {len(synced_data_no_images)}")
    
    save_data(global_synced_pose_data, episode_ends, output_dir)
    print("ROS bag processing complete. Data saved in:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_folder', type=str, default='/home/sam/bags/msl_bags/xarm_20_demos_2_5_2025',
                        help='Folder containing ROS bag files.')
    parser.add_argument('--output_dir', type=str, default='/home/sam/bags/msl_bags/rosbag_processed_data',
                        help='Output directory to save synchronized data.')
    parser.add_argument('--target_rate', type=float, default=10.0,
                        help='Target synchronization rate in Hz (i.e., reduced frequency).')
    args = parser.parse_args()
    
    process_rosbags(args.bags_folder, args.output_dir, args.target_rate)