import os
import pyarrow.parquet as pq
import json

DATASET_ROOT = "/bags/msl_bags/converted_groot_data"
CHUNK_DIR = os.path.join(DATASET_ROOT, "data/chunk-000")
META_DIR = os.path.join(DATASET_ROOT, "meta")
os.makedirs(META_DIR, exist_ok=True)

# Generate episodes.jsonl
episodes_path = os.path.join(META_DIR, "episodes.jsonl")
parquet_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".parquet")])

with open(episodes_path, "w") as f:
    for fname in parquet_files:
        full_path = os.path.join(CHUNK_DIR, fname)
        length = pq.read_table(full_path).num_rows
        episode_index = int(fname.replace("episode_", "").replace(".parquet", ""))
        f.write(json.dumps({
            "episode_index": episode_index,
            "tasks": ["Pick and Place"],
            "length": length
        }) + "\n")

print(f"episodes.jsonl written with {len(parquet_files)} entries.")

# Generate tasks.jsonl
tasks_path = os.path.join(META_DIR, "tasks.jsonl")
with open(tasks_path, "w") as f:
    f.write(json.dumps({
        "task_index": 0,
        "task": "Pick and Place"
    }) + "\n")

print("tasks.jsonl written.")
