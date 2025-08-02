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
import time
from scipy.spatial.transform import Rotation as R

# Install necessary packages
try:
    import zarr
    import imageio
except ImportError:
    print("Installing required packages...")
    os.system("pip install zarr imageio[ffmpeg]")







########################### IMPOTANT PARAMETERS ###########################
use_relative_action = True  # If True, compute relative action (delta position and orientation) instead of absolute pose
if use_relative_action:
    print("Using relative action (delta position and orientation) for the robot end effector.")
else:
    print("Using absolute pose for the robot end effector.")
response = input("Press Enter to continue...")
    
    





problematic_bags = []


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
        # response = input(f"Directory '{directory}' does not exist. Create it? (y/n): ")
        # if response.lower() == 'y':
        #     os.makedirs(directory)
        #     print(f"Directory '{directory}' created for data store.")
        # just print out the message
        print(f"Directory '{directory}' does not exist. Create it now...")

def get_closest_message(target_sec, messages, max_diff):
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

def synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, gripper_opening_msgs, target_rate, max_time_diff):
    """
    Synchronize pose and image messages by sampling at a period defined by target_rate.
    For each sample time, the function finds the closest pose message, fixed-camera image,
    and wrist-camera image.
    """
    T = 1.0 / target_rate
    if not pose_msgs or not fixed_msgs or not wrist_msgs or not gripper_opening_msgs:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("No messages found for one or more topics. Skipping this bag.")
        print("Check if the bag is missing some topics.")
        print("But you can continue to process the next bag...")
        # input("Press Enter to continue...")
        return [], []
    
    # Ensure the lists are sorted by time
    pose_msgs = sorted(pose_msgs, key=lambda x: x[2].to_sec())
    fixed_msgs = sorted(fixed_msgs, key=lambda x: x[2].to_sec())
    wrist_msgs = sorted(wrist_msgs, key=lambda x: x[2].to_sec())
    gripper_opening_msgs = sorted(gripper_opening_msgs, key=lambda x: x[2].to_sec())
    
    
    # Define the overlapping time window
    start_time = max(pose_msgs[0][2].to_sec(), fixed_msgs[0][2].to_sec(), wrist_msgs[0][2].to_sec(), gripper_opening_msgs[0][2].to_sec())
    end_time = min(pose_msgs[-1][2].to_sec(), fixed_msgs[-1][2].to_sec(), wrist_msgs[-1][2].to_sec(), gripper_opening_msgs[-1][2].to_sec())
    
    synced_data = []
    synced_data_no_images = []
    t_sample = start_time
    pose_entry, fixed_entry, wrist_entry, gripper_entry = None, None, None, None
    while t_sample <= end_time:
        pose_entry = get_closest_message(t_sample, pose_msgs, max_time_diff)
        fixed_entry = get_closest_message(t_sample, fixed_msgs, max_time_diff)
        wrist_entry = get_closest_message(t_sample, wrist_msgs, max_time_diff)
        
        if pose_entry is not None:
            # since gripper opening is recorded at the same time as pose, we can use pose time to sync gripper msg 
            best_pose_time = pose_entry[2].to_sec()
            gripper_entry = get_closest_message(best_pose_time, gripper_opening_msgs, max_time_diff)
        if pose_entry is None or fixed_entry is None or wrist_entry is None or gripper_entry is None:
            t_sample += T
            print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Skipping sample at time {t_sample} due to unable to sync messages.")
            continue
        synced_data.append({
            "timestamp": t_sample,
            "pose": pose_entry,
            "fixed": fixed_entry,
            "wrist": wrist_entry, 
            "gripper": gripper_entry
        })
        synced_data_no_images.append({
            "timestamp": t_sample,
            "pose": pose_entry,
            "gripper": gripper_entry
        })        
        t_sample += T
    return synced_data, synced_data_no_images
def save_synchronized_video(synced_data, output_dir, bag_index, bridge, target_rate, camera_type_name):
    print(f"Processing synchronized video for bag {bag_index} ({camera_type_name})...")
    fixed_camera_dir = os.path.join(output_dir, f"{bag_index:02d}")
    wrist_camera_dir = os.path.join(output_dir, f"{bag_index:02d}")
    ensure_dir(fixed_camera_dir)
    ensure_dir(wrist_camera_dir)

    fixed_camera_path = os.path.join(fixed_camera_dir, "3.mp4")
    wrist_camera_path = os.path.join(wrist_camera_dir, "1.mp4")

    fixed_writer = imageio.get_writer(fixed_camera_path, fps=target_rate, codec='libx264', format='FFMPEG')
    wrist_writer = imageio.get_writer(wrist_camera_path, fps=target_rate, codec='libx264', format='FFMPEG')
    printed = False
    
    # Set a fixed depth scale (in millimeters)
    DEPTH_MIN = 100    # 0.1 meters (200mm)
    DEPTH_MAX = 1500   # 2 meters (2000mm)

    for entry in synced_data:
        try:
            if 'depth' in camera_type_name:
                # print only once
                if not printed:
                    print(f"Processing depth image for bag {bag_index} ({camera_type_name})...")
                fixed_cv_image = bridge.imgmsg_to_cv2(entry["fixed"][1], "16UC1")
                wrist_cv_image = bridge.imgmsg_to_cv2(entry["wrist"][1], "16UC1")
                
                # Crop upper 1/3 and leftmost 1/4 for FIXED camera ONLY
                h, w = fixed_cv_image.shape
                crop_top = h // 6
                crop_left = w // 10
                fixed_cv_image_cropped = fixed_cv_image[crop_top:, crop_left:]
                if not printed:
                    print(f"Cropping the fixed camera by {crop_top} pixels from the top and {crop_left} pixels from the left.")
                    print(f"Fixed camera image shape after cropping: {fixed_cv_image_cropped.shape}")

                # Clip to fixed range, then normalize
                fixed_clipped = np.clip(fixed_cv_image_cropped, DEPTH_MIN, DEPTH_MAX)
                wrist_clipped = np.clip(wrist_cv_image, DEPTH_MIN, DEPTH_MAX)

                # Normalize to [0, 255]
                fixed_normalized = ((fixed_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255.0).astype(np.uint8)
                wrist_normalized = ((wrist_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255.0).astype(np.uint8)

                # Apply colormap (consistent across frames)
                fixed_rgb = cv2.applyColorMap(fixed_normalized, cv2.COLORMAP_JET)
                wrist_rgb = cv2.applyColorMap(wrist_normalized, cv2.COLORMAP_JET)


            elif 'infra' in camera_type_name:
                # print only once
                if not printed:
                    print(f"Processing infrared image for bag {bag_index} ({camera_type_name})...")
                # Infrared images (grayscale)
                fixed_cv_image = bridge.imgmsg_to_cv2(entry["fixed"][1], "mono8")
                wrist_cv_image = bridge.imgmsg_to_cv2(entry["wrist"][1], "mono8")
                fixed_rgb = cv2.cvtColor(fixed_cv_image, cv2.COLOR_GRAY2RGB)
                wrist_rgb = cv2.cvtColor(wrist_cv_image, cv2.COLOR_GRAY2RGB)

            else:  # color images (default)
                # print only once
                if not printed:
                    print(f"Processing color image for bag {bag_index} ({camera_type_name})...")
                fixed_cv_image = bridge.imgmsg_to_cv2(entry["fixed"][1], "bgr8")
                wrist_cv_image = bridge.imgmsg_to_cv2(entry["wrist"][1], "bgr8")
                fixed_rgb = cv2.cvtColor(fixed_cv_image, cv2.COLOR_BGR2RGB)
                wrist_rgb = cv2.cvtColor(wrist_cv_image, cv2.COLOR_BGR2RGB)

            printed = True
            fixed_writer.append_data(fixed_rgb)
            wrist_writer.append_data(wrist_rgb)

        except Exception as e:
            print(f"Error processing image for bag {bag_index}: {e}")

    fixed_writer.close()
    wrist_writer.close()
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
    
    
    
    
    # full pose representation
    # Original diffusion policy use action (~, 10) -- x, y, z, 6D orientation, gripper position
    # robot_eef_pos (~, 3) -- x, y, z
    # robot_eef_quat(~, 4) -- x, y, z, w
    # robot_gripper_qpos(~, 2) # left_finger, right_finger
    # timestamp(~, )
    

    data_store.create_dataset("action", shape=(0, 10), chunks=(chunk_size[0], 10), dtype=np.float64)
    data_store.create_dataset("timestamp", shape=(0,), chunks=chunk_size, dtype=np.float64)
    # data_store.create_dataset("robot_eef_pose", shape=(0, 6), chunks=(chunk_size[0], 6), dtype=np.float64)
    data_store.create_dataset("robot_eef_quat", shape=(0, 4), chunks=(chunk_size[0], 4), dtype=np.float64)
    # position
    data_store.create_dataset("robot_eef_pos", shape=(0, 3), chunks=(chunk_size[0], 3), dtype=np.float64)
    # end effector gripper position:
    data_store.create_dataset("robot_gripper_qpos", shape=(0, 2), chunks=(chunk_size[0], 2), dtype=np.float64)
    
    timestamps, position, orientation, gripper_pos, actions = [], [], [], [], []
        
    # Use synchronized pose data. Here, "action" is taken as the pose from the next time step.
    for i in range(len(synced_pose_data) - 1):  # Skip last sample for action
        _, msg, t = synced_pose_data[i]["pose"]
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
        
        # add gripper position
        _, gripper_msg, t_gripper = synced_pose_data[i]["gripper"]
        # record gripper position, left_finger, right_finger, each equal to the same value (which should range from 0-850 for xarm)
        gripper_pos.append([gripper_msg.data, gripper_msg.data])
      
        next_msg = synced_pose_data[i+1]["pose"][1]
        
        
        if use_relative_action:

            # Compute relative position
            delta_pos = np.array([
                next_msg.pose.position.x - msg.pose.position.x,
                next_msg.pose.position.y - msg.pose.position.y,
                next_msg.pose.position.z - msg.pose.position.z,
            ])

            # Compute relative orientation: q_rel = q_prev.inverse * q_current
            quat_prev = R.from_quat(quat)
            quat_next = R.from_quat([
                next_msg.pose.orientation.x,
                next_msg.pose.orientation.y,
                next_msg.pose.orientation.z,
                next_msg.pose.orientation.w
            ])
            quat_rel = quat_prev.inv() * quat_next

            # Convert relative quaternion to 6D
            orientation_6d = quat_to_six_d(quat_rel.as_quat())

            # Gripper binary action
            gripper_opened = 1 if gripper_msg.data > (800.0 / 2.0) else 0

            actions.append([
                delta_pos[0],
                delta_pos[1],
                delta_pos[2],
                orientation_6d[0],
                orientation_6d[1],
                orientation_6d[2],
                orientation_6d[3],
                orientation_6d[4],
                orientation_6d[5],
                gripper_opened
            ])
        
        else:
            # 6D orientation representation
            quat_next_msg = [
                next_msg.pose.orientation.x,
                next_msg.pose.orientation.y,
                next_msg.pose.orientation.z,
                next_msg.pose.orientation.w
            ]
            orientation_6d = quat_to_six_d(quat_next_msg)
            
            # action's last dim gripper opening = 1 if gripper_msg.data > 850 / 2, else 0
            gripper_opened = 1 if gripper_msg.data > (800.0 / 2.0) else 0
                
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
                gripper_opened
            ])
            
            
            
        # sanity check
        if t.to_sec() - t_gripper.to_sec() > 1.0 / 100.0: # if the difference is more than 10ms
            print(f"t: {t.to_sec()}, gripper t: {t_gripper.to_sec()}")
            raise Exception("Timestamps do not match.")
        # print(f"t: {t.to_sec()}, gripper: {gripper_msg.data}")
        # print(f"gripper_opened: {actions[-1]}")
    
    data_store["timestamp"].append(np.array(timestamps))
    data_store["robot_eef_pos"].append(np.array(position))
    data_store["action"].append(np.array(actions))
    data_store["robot_eef_quat"].append(np.array(orientation))
    data_store["robot_gripper_qpos"].append(np.array(gripper_pos))
    
    meta_store.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64), chunks=(len(episode_ends),))
    # print out episode ends and details on the data info
    print(f"Episode ends: {episode_ends}")
    print(f"Total samples: {len(timestamps)}")
    # total actions should be one less than total samples
    print(f"Total actions: {len(actions)}")
    # timestamps should be equal to total samples
    print(f"Total timestamps: {len(timestamps)}")
    print("Data and metadata saved successfully.")
    
    if len(problematic_bags) > 0:
        # print in red using ANSI Escape Codes not color
        print("\033[91mProblematic Bags: \033[0m")
        
        for bag in problematic_bags:
            print(f"\033[91m{bag}\033[0m")
        print("\033[91mPlease check the bags for missing topics or synchronization issues.\033[0m")
        print("\033[92mThose bags were automatically ignored so your processed data is still good, but you should check what happened to those bags...\033[0m")
        
def process_rosbags(bags_folder, output_dir, target_rate, image_topics_fixed_and_wrist_pair, max_time_diff):
    """
    For each ROS bag in the given folder, read and separate the robot pose and image messages,
    synchronize them based on the target_rate, write synchronized videos, and accumulate the
    synchronized pose data for saving into Zarr.
    """
    ensure_dir(output_dir, check=True)
    bag_files = [os.path.join(bags_folder, f) for f in os.listdir(bags_folder) if f.endswith(".bag")]
    # sort bag files by name
    bag_files.sort()
    # just for quick testing
    # bag_files = bag_files[37:42]
    
    global_synced_pose_data = []
    episode_ends = []
    total_samples = 0
    bridge = CvBridge()
    
    bag_index = 0
    for bag_file in bag_files:
        print(f"Processing bag : {bag_file}")
        bag = rosbag.Bag(bag_file, "r")
        
        pose_msgs = []
        fixed_msgs = []
        wrist_msgs = []
        gripper_opening_msgs = []
        
        for topic, msg, t in bag.read_messages():
            if topic == '/robot_end_effector_pose':
                pose_msgs.append((topic, msg, t))
            elif topic == '/robot_end_effector_opening':
                gripper_opening_msgs.append((topic, msg, t))
            elif topic == image_topics_fixed_and_wrist_pair[0]:
                fixed_msgs.append((topic, msg, t))
            elif topic == image_topics_fixed_and_wrist_pair[1]:
                wrist_msgs.append((topic, msg, t))
        
        bag.close()
        
        # Synchronize messages at the given target_rate.
        synced_data, synced_data_no_images = synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, gripper_opening_msgs, target_rate, max_time_diff)
        if synced_data == []:
            print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print in yellow using ANSI Escape Codes not color
            print(f"\033[93mNo synchronized data found for bag {bag_index}. Skipping this bag.\033[0m")
            problematic_bags.append(bag_file)
            continue
        
        save_synchronized_video(synced_data, output_dir, bag_index, bridge, target_rate, image_topics_fixed_and_wrist_pair[0])
        
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
    # add arguments for image topic fixed and wrist
    image_topics_fixed = ['/fixed_camera/color/image_raw']
    image_topics_wrist = ['/wrist_camera/color/image_raw']
    max_time_diff = 0.1 # seconds, tolerance for timestamp diff when syncing data between each two messages
    
    parser.add_argument('--bags_folder', type=str, default='/home/sam/bags/msl_bags/pick_and_place_NEW_4_20',
                        help='Folder containing ROS bag files.')
    parser.add_argument('--output_dir', type=str, default='/home/sam/bags/msl_bags/rosbag_processed_data',
                        help='Output directory to save synchronized data.')
    parser.add_argument('--target_rate', type=float, default=10.0,
                        help='Target synchronization rate in Hz (i.e., reduced frequency).')
    args = parser.parse_args()
    
    # image_topics_fixed = ['/fixed_camera/aligned_depth_to_color/image_raw', '/fixed_camera/color/image_raw', '/fixed_camera/infra1/image_rect_raw']
    # image_topics_wrist = ['/wrist_camera/aligned_depth_to_color/image_raw','/wrist_camera/color/image_raw',  '/wrist_camera/infra1/image_rect_raw']
    
    image_topics_fixed_and_wrist_pairs = {} # key is fixed and value is wrist, add it pair by pair
    for idx, fixed in enumerate(image_topics_fixed):
        wrist = image_topics_wrist[idx]
        # check they should be the same except for the topic prefix
        if fixed.split('/')[2:] == wrist.split('/')[2:]:
            image_topics_fixed_and_wrist_pairs[fixed] = wrist
            print(f"Image topics match: {fixed} and {wrist}, proceeding with synchronization.")
        else:
            raise Exception(f"Image topics do not match: {fixed} and {wrist}")
    
    
    
    for key, value in image_topics_fixed_and_wrist_pairs.items():
        print(f"Fixed camera topic: {key}, Wrist camera topic: {value}")
        image_topics_fixed_and_wrist_pair = (key, value)        
        # second element is the camera type name
        camera_type_name = key.split('/')[2]
        print(f"Camera type name: {camera_type_name}")
        final_output_dir = args.output_dir + f"_{camera_type_name}"
        process_rosbags(args.bags_folder, final_output_dir, args.target_rate, image_topics_fixed_and_wrist_pair, max_time_diff)
