import os
import json

# Set your dataset root
DATASET_ROOT = "/bags/msl_bags/converted_groot_data_absolute"
META_DIR = os.path.join(DATASET_ROOT, "meta")
os.makedirs(META_DIR, exist_ok=True)

# Define delta indices for 100-step horizon
delta_indices = list(range(100))

# Define modality config with mapping 
modality = {
    "state": {
        "single_arm": { "start": 0, "end": 9 },
        "gripper":    { "start": 9, "end": 10 }
    },
    "action": {
        "single_arm": {
            "start": 0,
            "end": 9,
            "delta_indices": delta_indices
        },
        "gripper": {
            "start": 9,
            "end": 10,
            "delta_indices": delta_indices
        }
    },
    "video": {
        "front": { "original_key": "observation.images.front" },
        "wrist": { "original_key": "observation.images.wrist" }
    },
    "annotation": {
        "human.task_description": { "original_key": "task_index" }
    },
    "mapping": {
        "single_arm": ["eef_position", "eef_orientation"],
        "gripper": ["gripper"]
    }
}

# Save to meta/modality.json
output_path = os.path.join(META_DIR, "modality.json")
with open(output_path, "w") as f:
    json.dump(modality, f, indent=2)

print(f"modality.json with mapping written to: {output_path}")
