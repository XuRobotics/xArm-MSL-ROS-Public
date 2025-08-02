"""
MIT License

Copyright (c) 2025 Xu Liu, Multi-Robot Systems Lab, Stanford University (assisted with ChatGPT and GitHub Copilot)

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
import cv2
from cv_bridge import CvBridge
import argparse
import time
from scipy.spatial.transform import Rotation as R

# Install necessary packages
try:
    import zarr  # (Not used for ACT, but left if needed elsewhere)
    import imageio  # (Not used for ACT here)
except ImportError:
    os.system("pip install zarr imageio[ffmpeg]")

# **MOD**: Ensure h5py is installed for HDF5 support
try:
    import h5py
except ImportError:
    print("Installing h5py for HDF5 support...")
    os.system("pip install h5py")
    import h5py

########################### IMPORTANT PARAMETERS ###########################
use_relative_action = False  # **MOD**: For ACT, we'll use absolute next pose as action (not deltas)
if use_relative_action:
    print("Using relative action (delta position and orientation) for end effector.")
else:
    print("Using absolute next pose for the robot end effector as action targets.")
# Remove interactive prompt for automated script execution:
# response = input("Press Enter to continue...")

problematic_bags = []

def ensure_dir(directory, check=False):
    """Create directory if it doesn't exist. If check=True and exists, prompt to delete."""
    if not check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created for data store.")
    elif os.path.exists(directory):
        response = input(f"Directory '{directory}' already exists. Delete it? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -r {directory}")
            os.makedirs(directory)
            print(f"Directory '{directory}' recreated (previous content deleted).")
        else:
            print("Exiting without overwriting existing data directory.")
            exit()
    else:
        os.makedirs(directory)
        print(f"Directory '{directory}' created for data store.")

def get_closest_message(target_sec, messages, max_diff):
    """
    Return the message from list closest in time to target_sec if within max_diff, else None.
    Each entry in messages is (topic, msg, t).
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

def synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, gripper_msgs, target_rate, max_time_diff):
    """
    Synchronize pose, gripper, and image messages by sampling at target_rate.
    Returns two lists:
      - synced_data: list of dicts with synchronized pose, fixed image, wrist image, gripper for each timestep.
      - synced_data_no_images: similar but without image data (for pose/gripper only).
    """
    if not pose_msgs or not fixed_msgs or not wrist_msgs or not gripper_msgs:
        print("+++ Missing messages for one or more topics in this bag, skipping. +++")
        return [], []
    # Sort messages by time
    pose_msgs.sort(key=lambda x: x[2].to_sec())
    fixed_msgs.sort(key=lambda x: x[2].to_sec())
    wrist_msgs.sort(key=lambda x: x[2].to_sec())
    gripper_msgs.sort(key=lambda x: x[2].to_sec())
    # Define overlapping time window across all streams
    start_time = max(pose_msgs[0][2].to_sec(), 
                     fixed_msgs[0][2].to_sec(), 
                     wrist_msgs[0][2].to_sec(),
                     gripper_msgs[0][2].to_sec())
    end_time   = min(pose_msgs[-1][2].to_sec(), 
                     fixed_msgs[-1][2].to_sec(), 
                     wrist_msgs[-1][2].to_sec(),
                     gripper_msgs[-1][2].to_sec())
    step = 1.0 / target_rate
    t_sample = start_time
    synced_data = []
    synced_data_no_images = []
    while t_sample <= end_time:
        pose_entry    = get_closest_message(t_sample, pose_msgs, max_time_diff)
        fixed_entry   = get_closest_message(t_sample, fixed_msgs, max_time_diff)
        wrist_entry   = get_closest_message(t_sample, wrist_msgs, max_time_diff)
        gripper_entry = None
        if pose_entry is not None:
            # Use pose timestamp to find closest gripper measurement (they should be nearly same time)
            pose_time = pose_entry[2].to_sec()
            gripper_entry = get_closest_message(pose_time, gripper_msgs, max_time_diff)
        if pose_entry is None or fixed_entry is None or wrist_entry is None or gripper_entry is None:
            # If any data is missing at this sample time, skip this timestamp
            t_sample += step
            continue
        synced_data.append({
            "timestamp": t_sample,
            "pose": pose_entry,         # tuple (topic, PoseStamped_msg, time)
            "fixed": fixed_entry,       # tuple (topic, Image_msg, time)
            "wrist": wrist_entry,       # tuple (topic, Image_msg, time)
            "gripper": gripper_entry    # tuple (topic, Float32_msg, time)
        })
        synced_data_no_images.append({
            "timestamp": t_sample,
            "pose": pose_entry,
            "gripper": gripper_entry
        })
        t_sample += step
    return synced_data, synced_data_no_images

def quat_to_six_d(quat):
    """
    Converts a quaternion [x, y, z, w] to a 6D rotation representation (first 2 columns of rotation matrix).
    """
    rot_matrix = R.from_quat(quat).as_matrix()
    rot_6d = rot_matrix[:, :2].flatten()
    return rot_6d

def process_rosbags(bags_folder, output_dir, target_rate, fixed_topic, wrist_topic, max_time_diff):
    """
    Process each ROS bag in the folder: synchronize data and save to HDF5 file per episode.
    """
    ensure_dir(output_dir, check=True)
    bag_files = [os.path.join(bags_folder, f) for f in os.listdir(bags_folder) if f.endswith(".bag")]
    bag_files.sort()  # sort by filename for consistent ordering
    bridge = CvBridge()
    global_index = 0  # episode index for naming files
    
    for bag_file in bag_files:
        print(f"Processing bag: {bag_file}")
        bag = rosbag.Bag(bag_file, "r")
        pose_msgs = []
        fixed_msgs = []
        wrist_msgs = []
        gripper_msgs = []
        # Read all messages and collect relevant topics
        for topic, msg, t in bag.read_messages():
            if topic == '/robot_end_effector_pose':
                pose_msgs.append((topic, msg, t))
            elif topic == '/robot_end_effector_opening':
                gripper_msgs.append((topic, msg, t))
            elif topic == fixed_topic:
                fixed_msgs.append((topic, msg, t))
            elif topic == wrist_topic:
                wrist_msgs.append((topic, msg, t))
        bag.close()
        
        # Synchronize messages at the target rate
        synced_data, synced_data_no_images = synchronize_messages(
            pose_msgs, fixed_msgs, wrist_msgs, gripper_msgs, target_rate, max_time_diff)
        if not synced_data:
            # No synchronized frames for this bag (or missing data), skip it
            print(f"Skipping bag {bag_file} due to insufficient synchronized data.")
            problematic_bags.append(bag_file)
            continue
        
        # Prepare HDF5 file for this episode
        episode_filename = os.path.join(output_dir, f"episode_{global_index:04d}.h5")
        with h5py.File(episode_filename, 'w') as h5f:
            T = len(synced_data)  # number of synchronized steps in this episode
            
            TARGET_RES = (640, 480)  # (width, height)

            # Resize early to ensure consistent shape
            first_fixed_img = bridge.imgmsg_to_cv2(synced_data[0]["fixed"][1], "bgr8")
            first_wrist_img = bridge.imgmsg_to_cv2(synced_data[0]["wrist"][1], "bgr8")
            first_fixed_rgb = cv2.cvtColor(first_fixed_img, cv2.COLOR_BGR2RGB)
            first_wrist_rgb = cv2.cvtColor(first_wrist_img, cv2.COLOR_BGR2RGB)

            first_fixed_rgb = cv2.resize(first_fixed_rgb, TARGET_RES)
            first_wrist_rgb = cv2.resize(first_wrist_rgb, TARGET_RES)

            Hf, Wf, _ = first_fixed_rgb.shape
            Hw, Ww, _ = first_wrist_rgb.shape

            # Create datasets for images (uint8)
            fixed_dset = h5f.create_dataset("fixed_images", shape=(T, Hf, Wf, 3), dtype=np.uint8)
            wrist_dset = h5f.create_dataset("wrist_images", shape=(T, Hw, Ww, 3), dtype=np.uint8)
            # Datasets for end-effector pose and gripper
            pos_dset   = h5f.create_dataset("ee_positions", shape=(T, 3), dtype=np.float32)
            ori_dset   = h5f.create_dataset("ee_orientations", shape=(T, 6), dtype=np.float32)
            grip_dset  = h5f.create_dataset("gripper_opening", shape=(T,), dtype=np.float32)
            # We will collect actions in a list first, then create dataset (T-1, 10)
            
            actions = []
            
            # Loop through synchronized data and fill datasets
            for i, entry in enumerate(synced_data):
                # Extract and store end-effector pose
                pose_msg = entry["pose"][1]  # geometry_msgs/PoseStamped (or Pose) message
                px = pose_msg.pose.position.x
                py = pose_msg.pose.position.y
                pz = pose_msg.pose.position.z
                quat = [
                    pose_msg.pose.orientation.x,
                    pose_msg.pose.orientation.y,
                    pose_msg.pose.orientation.z,
                    pose_msg.pose.orientation.w
                ]
                # Convert quaternion to 6D orientation representation
                ori_6d = quat_to_six_d(quat)
                pos_dset[i, :] = [px, py, pz]
                ori_dset[i, :] = ori_6d
                
                # Gripper opening value
                grip_value = entry["gripper"][1].data  # assuming Float32 message for gripper opening
                grip_dset[i] = grip_value
                
                # Convert and store RGB images
                fixed_img_bgr = bridge.imgmsg_to_cv2(entry["fixed"][1], "bgr8")
                wrist_img_bgr = bridge.imgmsg_to_cv2(entry["wrist"][1], "bgr8")
                fixed_img_rgb = cv2.cvtColor(fixed_img_bgr, cv2.COLOR_BGR2RGB)
                wrist_img_rgb = cv2.cvtColor(wrist_img_bgr, cv2.COLOR_BGR2RGB)

                fixed_img_rgb = cv2.resize(fixed_img_rgb, TARGET_RES)
                wrist_img_rgb = cv2.resize(wrist_img_rgb, TARGET_RES)

                assert fixed_img_rgb.dtype == np.uint8
                assert wrist_img_rgb.dtype == np.uint8

                fixed_dset[i, :] = fixed_img_rgb
                wrist_dset[i, :] = wrist_img_rgb
                
                # Compute action = next pose (pos, ori6d) + gripper open/close (binary), for all but last step
                if i < T - 1:
                    next_pose_msg = synced_data[i+1]["pose"][1]
                    # Next position and orientation
                    nxt_px = next_pose_msg.pose.position.x
                    nxt_py = next_pose_msg.pose.position.y
                    nxt_pz = next_pose_msg.pose.position.z
                    nxt_quat = [
                        next_pose_msg.pose.orientation.x,
                        next_pose_msg.pose.orientation.y,
                        next_pose_msg.pose.orientation.z,
                        next_pose_msg.pose.orientation.w
                    ]
                    nxt_ori6d = quat_to_six_d(nxt_quat)
                    # Binary gripper action (open=1 or close=0) based on current grip value
                    gripper_open_action = 1 if grip_value > 400.0 else 0  # threshold at half of ~800 max
                    actions.append([nxt_px, nxt_py, nxt_pz,
                                    nxt_ori6d[0], nxt_ori6d[1], nxt_ori6d[2],
                                    nxt_ori6d[3], nxt_ori6d[4], nxt_ori6d[5],
                                    gripper_open_action])
            # end for each timestep
            
            # Create and save the actions dataset
            if actions:
                action_arr = np.array(actions, dtype=np.float32)
                h5f.create_dataset("actions", data=action_arr, dtype=np.float32)
            else:
                # If for some reason there's no action (unlikely unless T=0), skip action dataset
                h5f.create_dataset("actions", shape=(0, 10), dtype=np.float32)
            
            # (Optional) Store metadata as HDF5 attributes
            h5f.attrs["episode_length"] = T
            h5f.attrs["fixed_topic"] = fixed_topic
            h5f.attrs["wrist_topic"] = wrist_topic
            print(f"Saved episode {global_index} with {T} frames: {episode_filename}")
        
        global_index += 1  # increment episode index
    # end for each bag
    
    # After processing all bags, report any problematic ones
    if problematic_bags:
        print("\nSome bags were skipped due to missing data or sync issues:")
        for pb in problematic_bags:
            print(f" - {pb}")
        print("Please check these bags for missing topics or timestamps misalignment.")
    else:
        print("\nAll bags processed successfully.")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_folder', type=str, default='/bags/msl_bags/IMPORTANT-distribution-pick-and-place-raw-bags-30',
                        help='Folder containing ROS bag files.')
    parser.add_argument('--output_dir', type=str, default='/bags/msl_bags/converted_act_data',
                        help='Output directory to save HDF5 episodes.')
    parser.add_argument('--target_rate', type=float, default=10.0,
                        help='Target synchronization rate in Hz.')
    parser.add_argument('--max_time_diff', type=float, default=0.1,
                        help='Max allowable time difference in seconds for message synchronization.')
    # Optionally allow specifying topics via args (else use defaults)
    parser.add_argument('--fixed_topic', type=str, default='/fixed_camera/color/image_raw',
                        help='ROS topic for the fixed camera images.')
    parser.add_argument('--wrist_topic', type=str, default='/wrist_camera/color/image_raw',
                        help='ROS topic for the wrist camera images.')
    args = parser.parse_args()
    
    # Print info about topics
    print(f"Fixed camera topic: {args.fixed_topic}, Wrist camera topic: {args.wrist_topic}")
    process_rosbags(args.bags_folder, args.output_dir, args.target_rate,
                    args.fixed_topic, args.wrist_topic, args.max_time_diff)