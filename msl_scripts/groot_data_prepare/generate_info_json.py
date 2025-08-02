import os
import json
import pyarrow.parquet as pq

# === CONFIG ===
DATASET_ROOT = "/bags/msl_bags/converted_groot_data"
CHUNK_DIR = os.path.join(DATASET_ROOT, "data/chunk-000")
META_DIR = os.path.join(DATASET_ROOT, "meta")
os.makedirs(META_DIR, exist_ok=True)

VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280
VIDEO_CHANNELS = 3
FPS = 10.0

# === Scan parquet files ===
parquet_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".parquet")])
total_episodes = len(parquet_files)
total_frames = sum(pq.read_table(os.path.join(CHUNK_DIR, f)).num_rows for f in parquet_files)

# === Build info.json ===
info = {
    "codebase_version": "v2.1",
    "robot_type": "xarm",
    "total_episodes": total_episodes,
    "total_frames": total_frames,
    "total_tasks": 1,
    "total_videos": total_episodes * 2,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": FPS,
    "splits": {
        "train": f"0:{total_episodes}"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [10],
            "names": [
                "delta_x", "delta_y", "delta_z",
                "orientation_0", "orientation_1", "orientation_2",
                "orientation_3", "orientation_4", "orientation_5",
                "gripper"
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [10],
            "names": [
                "orientation_0", "orientation_1", "orientation_2",
                "orientation_3", "orientation_4", "orientation_5",
                "pos_x", "pos_y", "pos_z",
                "gripper_state"
            ]
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": FPS,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
                "video.channels": VIDEO_CHANNELS
            }
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": FPS,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
                "video.channels": VIDEO_CHANNELS
            }
        },
        "timestamp": { "dtype": "float32", "shape": [1] },
        "frame_index": { "dtype": "int64", "shape": [1] },
        "episode_index": { "dtype": "int64", "shape": [1] },
        "index": { "dtype": "int64", "shape": [1] },
        "task_index": { "dtype": "int64", "shape": [1] }
    }
}

# === Write to disk ===
output_path = os.path.join(META_DIR, "info.json")
with open(output_path, "w") as f:
    json.dump(info, f, indent=2)

print(f"info.json written to: {output_path}")
print(f"Episodes: {total_episodes}, Frames: {total_frames}, Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
