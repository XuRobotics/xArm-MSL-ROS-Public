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
import imageio
import argparse
import time
from scipy.spatial.transform import Rotation as R
import pyarrow.parquet as pq


# Install necessary packages if not present
try:
    import imageio
except ImportError:
    print("Installing required packages...")
    os.system("pip install imageio[ffmpeg] pyarrow")

########################### IMPORTANT PARAMETERS ###########################
use_relative_action = True  # If True, compute relative (delta) actions; if False, use absolute pose as action.
if use_relative_action:
    print("Using relative action (delta position and orientation) for the robot end effector.")
else:
    print("Using absolute pose for the robot end effector.")
response = input("Press Enter to continue...")

problematic_bags = []

def ensure_dir(directory, check=False):
    """Create directory if it doesn't exist. If check=True and exists, confirm deletion."""
    if not check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
    elif os.path.exists(directory):
        response = input(f"Directory '{directory}' already exists. Delete it? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -r {directory}")
        else:
            print("Exiting without overwriting existing data directory.")
            exit()
        os.makedirs(directory)
    else:
        print(f"Directory '{directory}' does not exist. Creating it...")
        os.makedirs(directory)

def get_closest_message(target_sec, messages, max_diff):
    """
    Given a target time (seconds) and a list of (topic, msg, time) tuples,
    return the tuple with timestamp closest to target_sec (within max_diff). 
    Returns None if no message is within max_diff.
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
    Synchronize pose, fixed-camera image, wrist-camera image, and gripper opening messages at a given target rate.
    Returns two lists:
      - synced_data: list of dicts with keys "timestamp", "pose", "fixed", "wrist", "gripper" for each synchronized timestep.
      - synced_data_no_images: similar list without the image data (only pose and gripper), for faster processing of numeric data.
    """
    if not pose_msgs or not fixed_msgs or not wrist_msgs or not gripper_msgs:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Missing one or more data streams in this bag; skipping synchronization.")
        return [], []
    # Sort messages by timestamp
    pose_msgs.sort(key=lambda x: x[2].to_sec())
    fixed_msgs.sort(key=lambda x: x[2].to_sec())
    wrist_msgs.sort(key=lambda x: x[2].to_sec())
    gripper_msgs.sort(key=lambda x: x[2].to_sec())
    # Determine overlapping time window for all sensors
    start_time = max(pose_msgs[0][2].to_sec(), fixed_msgs[0][2].to_sec(),
                     wrist_msgs[0][2].to_sec(), gripper_msgs[0][2].to_sec())
    end_time = min(pose_msgs[-1][2].to_sec(), fixed_msgs[-1][2].to_sec(),
                   wrist_msgs[-1][2].to_sec(), gripper_msgs[-1][2].to_sec())
    T = 1.0 / target_rate
    synced_data = []
    synced_data_no_images = []
    t_sample = start_time
    while t_sample <= end_time:
        pose_entry = get_closest_message(t_sample, pose_msgs, max_time_diff)
        fixed_entry = get_closest_message(t_sample, fixed_msgs, max_time_diff)
        wrist_entry = get_closest_message(t_sample, wrist_msgs, max_time_diff)
        gripper_entry = None
        if pose_entry is not None:
            # Use pose timestamp to find closest gripper state (they should have identical times in our data)
            pose_time = pose_entry[2].to_sec()
            gripper_entry = get_closest_message(pose_time, gripper_msgs, max_time_diff)
        if pose_entry is None or fixed_entry is None or wrist_entry is None or gripper_entry is None:
            # Skip this time if any sensor is unsynchronized beyond tolerance
            t_sample += T
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

def quat_to_six_d(quat):
    """
    Convert a quaternion [x, y, z, w] to a 6D rotation representation 
    (first two columns of the 3x3 rotation matrix, flattened to length 6).
    """
    rot_matrix = R.from_quat(quat).as_matrix()
    rot_6d = rot_matrix[:, :2].flatten()
    return rot_6d

def save_synchronized_video(synced_data, output_dir, episode_index, bridge, target_rate, camera_type_name):
    """
    Create synchronized videos for the fixed and wrist cameras for a given episode.
    Returns the image shape for each camera and computed image stats.
    """
    print(f"Processing synchronized video for episode {episode_index} ({camera_type_name})...")
    # Prepare output video paths
    fixed_cam_dir = os.path.join(output_dir, "videos", "chunk-000", "observation.images.front")
    wrist_cam_dir = os.path.join(output_dir, "videos", "chunk-000", "observation.images.wrist")
    ensure_dir(fixed_cam_dir)
    ensure_dir(wrist_cam_dir)
    fixed_video_path = os.path.join(fixed_cam_dir, f"episode_{episode_index:06d}.mp4")
    wrist_video_path = os.path.join(wrist_cam_dir, f"episode_{episode_index:06d}.mp4")
    fixed_writer = imageio.get_writer(fixed_video_path, fps=target_rate, codec='libx264', format='FFMPEG')
    wrist_writer = imageio.get_writer(wrist_video_path, fps=target_rate, codec='libx264', format='FFMPEG')
    printed = False
    shape_fixed = None
    shape_wrist = None
    # Initialize accumulators for image stats (per channel)
    fixed_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    fixed_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    fixed_sum = np.zeros(3, dtype=np.float64)
    fixed_sum_sq = np.zeros(3, dtype=np.float64)
    wrist_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    wrist_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    wrist_sum = np.zeros(3, dtype=np.float64)
    wrist_sum_sq = np.zeros(3, dtype=np.float64)
    frame_count = 0
    for entry in synced_data:
        try:
            # Handle different image types
            if 'depth' in camera_type_name:  # Depth images
                if not printed:
                    print(f"Processing depth images for episode {episode_index}...")
                fixed_cv = bridge.imgmsg_to_cv2(entry["fixed"][1], "16UC1")
                wrist_cv = bridge.imgmsg_to_cv2(entry["wrist"][1], "16UC1")
                # Crop fixed cam depth (crop top 1/6 height and left 1/10 width as in original code)
                h, w = fixed_cv.shape
                fixed_cv = fixed_cv[h//6:, w//10:]
                if not printed:
                    print(f"Fixed depth image cropped to {fixed_cv.shape}")
                # Depth normalization (0.1m to 1.5m mapped to 0-255)
                DEPTH_MIN, DEPTH_MAX = 100, 1500  # in millimeters
                fixed_clip = np.clip(fixed_cv, DEPTH_MIN, DEPTH_MAX)
                wrist_clip = np.clip(wrist_cv, DEPTH_MIN, DEPTH_MAX)
                fixed_norm = ((fixed_clip - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255.0).astype(np.uint8)
                wrist_norm = ((wrist_clip - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255.0).astype(np.uint8)
                # Apply color mapping to depth (JET colormap)
                fixed_rgb = cv2.applyColorMap(fixed_norm, cv2.COLORMAP_JET)
                wrist_rgb = cv2.applyColorMap(wrist_norm, cv2.COLORMAP_JET)
                raise NotImplementedError("Infrared images are not yet tested for GR00T model, may need resize.")
            elif 'infra' in camera_type_name:  # Infrared images (mono8)
                if not printed:
                    print(f"Processing infrared images for episode {episode_index}...")
                fixed_cv = bridge.imgmsg_to_cv2(entry["fixed"][1], "mono8")
                wrist_cv = bridge.imgmsg_to_cv2(entry["wrist"][1], "mono8")
                fixed_rgb = cv2.cvtColor(fixed_cv, cv2.COLOR_GRAY2RGB)
                wrist_rgb = cv2.cvtColor(wrist_cv, cv2.COLOR_GRAY2RGB)
                raise NotImplementedError("Infrared images are not yet tested for GR00T model, may need resize.")
            else:  # Color images (bgr8)
                if not printed:
                    print(f"Processing color images for episode {episode_index}...")
                fixed_cv = bridge.imgmsg_to_cv2(entry["fixed"][1], "bgr8")
                wrist_cv = bridge.imgmsg_to_cv2(entry["wrist"][1], "bgr8")
                fixed_rgb = cv2.cvtColor(fixed_cv, cv2.COLOR_BGR2RGB)
                wrist_rgb = cv2.cvtColor(wrist_cv, cv2.COLOR_BGR2RGB)
                # Resize to match GR00T's default resolution (1280x720)
                fixed_rgb = cv2.resize(fixed_rgb, (1280, 720))
                wrist_rgb = cv2.resize(wrist_rgb, (1280, 720))
            printed = True
            # Append frames to videos
            fixed_writer.append_data(fixed_rgb)
            wrist_writer.append_data(wrist_rgb)
            # Capture image shape (height, width, channels)
            if shape_fixed is None:
                shape_fixed = fixed_rgb.shape  # e.g. (480, 640, 3)
            if shape_wrist is None:
                shape_wrist = wrist_rgb.shape
            # Update image stats for normalization (treat each pixel as sample)
            # Normalize pixel values to [0,1] for stats calculation
            fixed_float = fixed_rgb.astype(np.float32) / 255.0
            wrist_float = wrist_rgb.astype(np.float32) / 255.0
            # Per-frame min/max per channel
            frame_fixed_min = fixed_float.min(axis=(0, 1))
            frame_fixed_max = fixed_float.max(axis=(0, 1))
            frame_wrist_min = wrist_float.min(axis=(0, 1))
            frame_wrist_max = wrist_float.max(axis=(0, 1))
            fixed_min = np.minimum(fixed_min, frame_fixed_min)
            fixed_max = np.maximum(fixed_max, frame_fixed_max)
            wrist_min = np.minimum(wrist_min, frame_wrist_min)
            wrist_max = np.maximum(wrist_max, frame_wrist_max)
            # Accumulate sum and sum of squares for mean/std
            fixed_sum += fixed_float.sum(axis=(0, 1))
            fixed_sum_sq += (fixed_float ** 2).sum(axis=(0, 1))
            wrist_sum += wrist_float.sum(axis=(0, 1))
            wrist_sum_sq += (wrist_float ** 2).sum(axis=(0, 1))
            frame_count += 1
        except Exception as e:
            print(f"Error processing image frame for episode {episode_index}: {e}")
    fixed_writer.close()
    wrist_writer.close()
    print(f"Finished writing videos for episode {episode_index}.")
    # Compute per-channel mean and std over all pixels of all frames
    if frame_count > 0 and shape_fixed is not None and shape_wrist is not None:
        Hf, Wf, Cf = shape_fixed  # fixed camera resolution
        total_pix_fixed = frame_count * Hf * Wf
        fixed_mean = fixed_sum / total_pix_fixed
        fixed_var = (fixed_sum_sq / total_pix_fixed) - (fixed_mean ** 2)
        fixed_std = np.sqrt(np.clip(fixed_var, 0, None))
        Hw, Ww, Cw = shape_wrist  # wrist camera resolution
        total_pix_wrist = frame_count * Hw * Ww
        wrist_mean = wrist_sum / total_pix_wrist
        wrist_var = (wrist_sum_sq / total_pix_wrist) - (wrist_mean ** 2)
        wrist_std = np.sqrt(np.clip(wrist_var, 0, None))
    else:
        fixed_mean = fixed_std = None
        wrist_mean = wrist_std = None
    # Structure image stats to return
    image_stats = {
        "front": {
            "min": fixed_min.tolist(),
            "max": fixed_max.tolist(),
            "mean": fixed_mean.tolist() if fixed_mean is not None else [0.0, 0.0, 0.0],
            "std": fixed_std.tolist() if fixed_std is not None else [0.0, 0.0, 0.0],
            "count": frame_count
        },
        "wrist": {
            "min": wrist_min.tolist(),
            "max": wrist_max.tolist(),
            "mean": wrist_mean.tolist() if wrist_mean is not None else [0.0, 0.0, 0.0],
            "std": wrist_std.tolist() if wrist_std is not None else [0.0, 0.0, 0.0],
            "count": frame_count
        }
    }
    return shape_fixed, shape_wrist, image_stats

def process_episode_to_parquet(synced_pose_data, episode_index, output_data_dir, use_relative_action=True, base_index=0):
    """
    Convert synchronized pose (and gripper) data for one episode into a Parquet file.
    Returns the number of samples and lists of states/actions for stats.
    """
    # Prepare arrays for data
    timestamps = []
    positions = []
    quaternions = []
    gripper_values = []
    # Extract data from synced_pose_data
    for entry in synced_pose_data:
        _, pose_msg, t = entry["pose"]
        _, gripper_msg, tg = entry["gripper"]
        timestamps.append(t.to_sec())
        # Robot end-effector position
        positions.append(np.array([pose_msg.pose.position.x,
                                   pose_msg.pose.position.y,
                                   pose_msg.pose.position.z], dtype=np.float64))
        # End-effector orientation quaternion
        quat = np.array([pose_msg.pose.orientation.x,
                         pose_msg.pose.orientation.y,
                         pose_msg.pose.orientation.z,
                         pose_msg.pose.orientation.w], dtype=np.float64)
        quaternions.append(quat)
        # Gripper opening (normalize 0-1)
        grip_norm = max(0.0, min(1.0, gripper_msg.data / 850.0))
        gripper_values.append(grip_norm)
    num_samples = len(synced_pose_data)
    # Compute state vectors (6D orientation + position + gripper)
    state_vectors = []
    for i in range(num_samples):
        rot6d = quat_to_six_d(quaternions[i]).astype(np.float32)
        pos = positions[i].astype(np.float32)
        grip = np.array([gripper_values[i]], dtype=np.float32)
        state_vec = np.concatenate([rot6d, pos, grip]).astype(np.float32)
        state_vectors.append(state_vec)
    # Compute action vectors
    action_vectors = []
    for i in range(num_samples):
        if i < num_samples - 1:
            if use_relative_action:
                # Delta to next state
                delta_pos = (positions[i+1] - positions[i]).astype(np.float32)
                quat_i = R.from_quat(quaternions[i])
                quat_next = R.from_quat(quaternions[i+1])
                quat_rel = quat_i.inv() * quat_next  # relative rotation from i to i+1
                orient_6d = quat_to_six_d(quat_rel.as_quat()).astype(np.float32)
                grip_cmd = np.array([1 if gripper_values[i] > 0.5 else 0], dtype=np.float32)
                action_vec = np.concatenate([delta_pos, orient_6d, grip_cmd])
            else:
                # Absolute next pose as action
                next_pos = positions[i+1].astype(np.float32)
                next_orient6d = quat_to_six_d(quaternions[i+1]).astype(np.float32)
                grip_cmd = np.array([1 if gripper_values[i] > 0.5 else 0], dtype=np.float32)
                action_vec = np.concatenate([next_pos, next_orient6d, grip_cmd])
        else:
            # For last frame, action = zero movement (hold last pose)
            if use_relative_action:
                zero_pos = np.zeros(3, dtype=np.float32)
                identity_quat = np.array([0, 0, 0, 1], dtype=np.float32)
                orient_6d = quat_to_six_d(identity_quat).astype(np.float32)
                grip_cmd = np.array([1 if gripper_values[i] > 0.5 else 0], dtype=np.float32)
                action_vec = np.concatenate([zero_pos, orient_6d, grip_cmd])
            else:
                hold_pos = positions[i].astype(np.float32)
                hold_orient6d = quat_to_six_d(quaternions[i]).astype(np.float32)
                grip_cmd = np.array([1 if gripper_values[i] > 0.5 else 0], dtype=np.float32)
                action_vec = np.concatenate([hold_pos, hold_orient6d, grip_cmd])
        action_vectors.append(action_vec.astype(np.float32))
    # Prepare index columns
    frame_indices = np.arange(num_samples, dtype=np.int64)
    episode_indices = np.full(num_samples, episode_index, dtype=np.int64)
    global_indices = frame_indices + base_index
    task_indices = np.zeros(num_samples, dtype=np.int64)  # single task index 0 for all
    # Build PyArrow table for Parquet
    import pyarrow as pa
    data = {
        "timestamp": np.array(timestamps, dtype=np.float32),
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "index": global_indices,
        "task_index": task_indices,
        "observation.state": pa.array([vec.tolist() for vec in state_vectors], type=pa.list_(pa.float32())),
        "action": pa.array([vec.tolist() for vec in action_vectors], type=pa.list_(pa.float32()))
    }
    table = pa.Table.from_pydict(data)
    output_file = os.path.join(output_data_dir, f"episode_{episode_index:06d}.parquet")
    # Ensure output directory exists
    ensure_dir(output_data_dir)
    # Write table to Parquet file
    pq.write_table(table, output_file, version="2.6", compression="snappy")
    return num_samples, state_vectors, action_vectors, timestamps, frame_indices.tolist(), episode_indices.tolist(), task_indices.tolist()

def process_rosbags(bags_folder, output_dir, target_rate=10.0, image_topics_fixed_and_wrist_pair=('/fixed_camera/color/image_raw', '/wrist_camera/color/image_raw'), max_time_diff=0.1):
    """
    Process all ROS bag files in a folder and convert them to the GR00T fine-tuning dataset format.
    """
    ensure_dir(output_dir, check=True)
    bag_files = [os.path.join(bags_folder, f) for f in os.listdir(bags_folder) if f.endswith(".bag")]
    bag_files.sort()
    bridge = CvBridge()
    episode_lengths = []
    total_frames = 0
    total_episodes = 0
    first_shape_front = None
    first_shape_wrist = None
    image_stats_all = []
    global_index_offset = 0
    # Loop through each bag (episode)
    for bag_index, bag_file in enumerate(bag_files):
        print(f"\nProcessing bag file: {bag_file}")
        bag = rosbag.Bag(bag_file, "r")
        pose_msgs, fixed_msgs, wrist_msgs, gripper_msgs = [], [], [], []
        # Read all messages from the bag
        for topic, msg, t in bag.read_messages():
            if topic == '/robot_end_effector_pose':    # geometry_msgs/PoseStamped
                pose_msgs.append((topic, msg, t))
            elif topic == '/robot_end_effector_opening':  # std_msgs/Float32 (or similar gripper position)
                gripper_msgs.append((topic, msg, t))
            elif topic == image_topics_fixed_and_wrist_pair[0]:
                fixed_msgs.append((topic, msg, t))
            elif topic == image_topics_fixed_and_wrist_pair[1]:
                wrist_msgs.append((topic, msg, t))
        bag.close()
        # Synchronize messages to target rate
        synced_data, synced_data_no_images = synchronize_messages(pose_msgs, fixed_msgs, wrist_msgs, gripper_msgs, target_rate, max_time_diff)
        if not synced_data:
            print(f"Skipping bag {bag_index}: unable to sync all topics (possibly missing data).")
            problematic_bags.append(bag_file)
            continue
        # Save videos for this episode
        shape_front, shape_wrist, image_stats = save_synchronized_video(synced_data, output_dir, bag_index, bridge, target_rate, image_topics_fixed_and_wrist_pair[0])
        if first_shape_front is None and shape_front is not None:
            first_shape_front = shape_front
        if first_shape_wrist is None and shape_wrist is not None:
            first_shape_wrist = shape_wrist
        image_stats_all.append({"episode_index": bag_index, "front": image_stats["front"], "wrist": image_stats["wrist"]})
        # Save pose/action data to Parquet
        num_samples, state_vecs, action_vecs, timestamps, frame_idx, ep_idx, task_idx = process_episode_to_parquet(
            synced_data_no_images, bag_index, os.path.join(output_dir, "data", "chunk-000"),
            use_relative_action=use_relative_action, base_index=global_index_offset)
        episode_lengths.append(num_samples)
        total_frames += num_samples
        global_index_offset += num_samples
        total_episodes += 1
        print(f"Episode {bag_index}: {num_samples} frames synchronized and saved.")
    # Generate meta files in output_dir/meta
    meta_dir = os.path.join(output_dir, "meta")
    ensure_dir(meta_dir)
    # episodes.jsonl
    import json
    episodes_list = []
    for i, length in enumerate(episode_lengths):
        episodes_list.append({"episode_index": i, "tasks": ["Pick and Place"], "length": length})
    with open(os.path.join(meta_dir, "episodes.jsonl"), 'w') as f:
        for entry in episodes_list:
            f.write(json.dumps(entry) + "\n")
    # tasks.jsonl
    tasks_list = [{"task_index": 0, "task": "Pick and Place"}]
    with open(os.path.join(meta_dir, "tasks.jsonl"), 'w') as f:
        for entry in tasks_list:
            f.write(json.dumps(entry) + "\n")
    # info.json
    if first_shape_front is None:
        first_shape_front = (0, 0, 0)
    if first_shape_wrist is None:
        first_shape_wrist = (0, 0, 0)
    hf, wf, cf = first_shape_front
    hw, ww, cw = first_shape_wrist
    info = {
        "codebase_version": "v2.1",
        "robot_type": "xarm",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * 2,
        "total_chunks": 1 if total_episodes < 1000 else (total_episodes // 1000),
        "chunks_size": 1000,
        "fps": target_rate,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [len(action_vecs[0]) if total_episodes > 0 else 0],
                "names": (
                    ["delta_x", "delta_y", "delta_z"] + [f"orientation_{j}" for j in range(6)] + ["gripper"]
                    if use_relative_action else
                    ["x", "y", "z"] + [f"orientation_{j}" for j in range(6)] + ["gripper"]
                )
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [len(state_vecs[0]) if total_episodes > 0 else 0],
                "names": [f"orientation_state_{j}" for j in range(6)] + ["pos_x", "pos_y", "pos_z", "gripper_state"]
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [hf, wf, cf],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(target_rate),
                    "video.height": hf,
                    "video.width": wf,
                    "video.channels": cf,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [hw, ww, cw],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(target_rate),
                    "video.height": hw,
                    "video.width": ww,
                    "video.channels": cw,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        }
    }
    with open(os.path.join(meta_dir, "info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    # modality.json (defines structure of state/action vectors and mapping of modalities)
    modality = {
        "state": {
            "eef_orientation": {"start": 0, "end": 6, "rotation_type": "6d"},
            "eef_position": {"start": 6, "end": 9},
            "gripper": {"start": 9, "end": 10}
        },
        "action": {
            "eef_position": {"start": 0, "end": 3, "absolute": False if use_relative_action else True},
            "eef_orientation": {"start": 3, "end": 9, "rotation_type": "6d", "absolute": False if use_relative_action else True},
            "gripper": {"start": 9, "end": 10, "absolute": True}
        },
        "video": {
            "front": {"original_key": "observation.images.front"},
            "wrist": {"original_key": "observation.images.wrist"}
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        }
    }
    with open(os.path.join(meta_dir, "modality.json"), 'w') as f:
        json.dump(modality, f, indent=2)
    # episodes_stats.jsonl (statistics per episode for each field)
    episodes_stats = []
    global_index_offset = 0
    import pyarrow.parquet as pq
    for i, length in enumerate(episode_lengths):
        ep_file = os.path.join(output_dir, "data", "chunk-000", f"episode_{i:06d}.parquet")
        table = pq.read_table(ep_file)
        data = table.to_pydict()
        # Convert list fields to numpy arrays for stat calculations
        action_arr = np.stack([np.array(x) for x in data["action"]], axis=0)
        state_arr = np.stack([np.array(x) for x in data["observation.state"]], axis=0)
        timestamp_arr = np.array(data["timestamp"], dtype=np.float32)
        frame_idx_arr = np.array(data["frame_index"], dtype=np.int64)
        ep_idx_arr = np.array(data["episode_index"], dtype=np.int64)
        task_idx_arr = np.array(data["task_index"], dtype=np.int64)
        # Global index values for this episode
        global_idx_arr = np.arange(length, dtype=np.int64) + global_index_offset
        global_index_offset += length
        # Helper to compute stats for a 2D array column-wise or 1D
        def compute_stats(arr):
            if arr.ndim == 1:
                return {
                    "min": [float(arr.min())],
                    "max": [float(arr.max())],
                    "mean": [float(arr.mean())],
                    "std": [float(arr.std())],
                    "count": [arr.shape[0]]
                }
            else:
                return {
                    "min": arr.min(axis=0).tolist(),
                    "max": arr.max(axis=0).tolist(),
                    "mean": arr.mean(axis=0).tolist(),
                    "std": arr.std(axis=0).tolist(),
                    "count": [arr.shape[0]]
                }
        # Get image stats from earlier recorded data
        img_stats = next((item for item in image_stats_all if item["episode_index"] == i), None)
        front_stats = {}
        wrist_stats = {}
        if img_stats:
            # Format image stats (min, max, mean, std as 3x1x1 nested lists per channel)
            front_stats = {
                "min": [[[img_stats["front"]["min"][0]]], [[img_stats["front"]["min"][1]]], [[img_stats["front"]["min"][2]]]],
                "max": [[[img_stats["front"]["max"][0]]], [[img_stats["front"]["max"][1]]], [[img_stats["front"]["max"][2]]]],
                "mean": [[[img_stats["front"]["mean"][0]]], [[img_stats["front"]["mean"][1]]], [[img_stats["front"]["mean"][2]]]],
                "std": [[[img_stats["front"]["std"][0]]], [[img_stats["front"]["std"][1]]], [[img_stats["front"]["std"][2]]]],
                "count": [img_stats["front"]["count"]]
            }
            wrist_stats = {
                "min": [[[img_stats["wrist"]["min"][0]]], [[img_stats["wrist"]["min"][1]]], [[img_stats["wrist"]["min"][2]]]],
                "max": [[[img_stats["wrist"]["max"][0]]], [[img_stats["wrist"]["max"][1]]], [[img_stats["wrist"]["max"][2]]]],
                "mean": [[[img_stats["wrist"]["mean"][0]]], [[img_stats["wrist"]["mean"][1]]], [[img_stats["wrist"]["mean"][2]]]],
                "std": [[[img_stats["wrist"]["std"][0]]], [[img_stats["wrist"]["std"][1]]], [[img_stats["wrist"]["std"][2]]]],
                "count": [img_stats["wrist"]["count"]]
            }
        # Compile stats for this episode
        ep_stats = {
            "episode_index": i,
            "stats": {
                "action": compute_stats(action_arr),
                "observation.state": compute_stats(state_arr),
                "observation.images.front": front_stats,
                "observation.images.wrist": wrist_stats,
                "timestamp": compute_stats(timestamp_arr),
                "frame_index": compute_stats(frame_idx_arr),
                "episode_index": compute_stats(ep_idx_arr),
                "index": compute_stats(global_idx_arr),
                "task_index": compute_stats(task_idx_arr)
            }
        }
        episodes_stats.append(ep_stats)
    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), 'w') as f:
        for entry in episodes_stats:
            f.write(json.dumps(entry) + "\n")
    # Report any skipped bags
    if problematic_bags:
        print("\nThe following bags were skipped due to missing data or sync issues:")
        for pb in problematic_bags:
            print(f" - {pb}")
    print(f"\nConversion complete. Dataset saved to '{output_dir}'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_folder', type=str, default='/bags/msl_bags/IMPORTANT-distribution-pick-and-place-raw-bags-30',
                        help='Folder containing ROS bag files.')
    parser.add_argument('--output_dir', type=str, default='/bags/msl_bags/converted_groot_data',
                        help='Output directory to save the converted dataset.')
    parser.add_argument('--target_rate', type=float, default=10.0,
                        help='Target synchronization rate (Hz) for downsampling.')
    args = parser.parse_args()
    image_topics_fixed = ['/fixed_camera/color/image_raw']
    image_topics_wrist = ['/wrist_camera/color/image_raw']
    # (Assumes corresponding fixed/wrist topics share the same suffix after the camera name)
    image_topics_pair = (image_topics_fixed[0], image_topics_wrist[0])
    process_rosbags(args.bags_folder, args.output_dir, target_rate=args.target_rate,
                    image_topics_fixed_and_wrist_pair=image_topics_pair, max_time_diff=0.1)
