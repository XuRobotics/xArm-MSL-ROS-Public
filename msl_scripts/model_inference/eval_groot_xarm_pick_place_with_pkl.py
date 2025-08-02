import os
import time
import numpy as np
import cv2
import torch
import json
from torchvision import transforms

import sys
sys.path.append("/home/xarm/Isaac-GR00T")

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

np.set_printoptions(precision=3, suppress=True, linewidth=200, threshold=10000)


# === CONFIG ===
CHECKPOINT_PATH = "/home/xarm/bags/msl_bags/groot_checkpoints/xarm_pick_place_absolute_pose_run6_batch_16_horizon_100/checkpoint-23000"
MODALITY_JSON = "/home/xarm/bags/msl_bags/converted_groot_data_absolute/meta/modality.json"
IMAGE_DIR = "/home/xarm/xArm-MSL-ROS/msl_scripts/image_frames_extracted/inference_data_in_dis"
FRAME_COUNT = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_INSTRUCTION = "Pick and place the object."


# === Load modality config ===
def load_modality_config(modality_json_path):
    with open(modality_json_path, "r") as f:
        modality_dict = json.load(f)

    return {
        "video": ModalityConfig(delta_indices=[-1, 0], modality_keys=[
            modality_dict["video"]["front"]["original_key"],
            modality_dict["video"]["wrist"]["original_key"]
        ]),
        "state": ModalityConfig(delta_indices=[-1, 0], modality_keys=["observation.state"]),
        "action": ModalityConfig(delta_indices=[0], modality_keys=["action"]),
        "language": ModalityConfig(delta_indices=[-1, 0], modality_keys=[
            modality_dict["annotation"]["human.task_description"]["original_key"]
        ])
    }


# === Preprocessing ===
def load_and_preprocess(img_path, size=(1280, 720)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img  # return raw uint8 numpy array


# === Hardcoded initial state (from your earlier message) ===
INIT_STATE_VECTOR = [
    -0.9996057, 0.00008499, 0.00018754,
    0.9999933, 0.02808065, -0.00365329,
    259.1312, 2.76629, 258.1401,
    0.9282353
]
# INIT_STATE = torch.tensor([[INIT_STATE_VECTOR]], dtype=torch.float32)  # stay on CPU


if __name__ == "__main__":
    modality_config = load_modality_config(MODALITY_JSON)
    # modality_config["action"].delta_indices = list(range(100))
    if modality_config["action"].delta_indices != list(range(200)):
        raise ValueError("Expected action delta indices to be 0-199 for 200-step horizon, DEBUG THIS!")
    transform_config = DATA_CONFIG_MAP["xarm_dualcam"].transform()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = Gr00tPolicy(
        model_path=CHECKPOINT_PATH,
        modality_config=modality_config,
        modality_transform=transform_config,
        embodiment_tag=EmbodimentTag.OXE_DROID,
        device=DEVICE
    )
    policy.model.to(DEVICE)

    # Force submodules to CUDA
    policy.model.action_head.to(DEVICE)

    print(f"Running inference on {FRAME_COUNT} image pairs from: {IMAGE_DIR}")
    actions = []

    for i in range(FRAME_COUNT):
        front_path = os.path.join(IMAGE_DIR, f"camera_3_frame_{i:04d}.png")
        wrist_path = os.path.join(IMAGE_DIR, f"camera_1_frame_{i:04d}.png")

        if not os.path.exists(front_path) or not os.path.exists(wrist_path):
            print(f"[{i:04d}] Missing image pair")
            continue

        # Load images as raw NumPy arrays (uint8)
        front_img = load_and_preprocess(front_path)   # shape (H, W, 3), dtype uint8
        wrist_img = load_and_preprocess(wrist_path)

        front_img = front_img[np.newaxis, np.newaxis, ...]  # shape (1, 1, H, W, 3)
        wrist_img = wrist_img[np.newaxis, np.newaxis, ...]

        INIT_STATE = np.array([[INIT_STATE_VECTOR]], dtype=np.float32)  # shape (1, 1, 10)

        # Prepare input dictionary WITHOUT torch conversion
        step_data = {
            "video.front": front_img,                     # still np.uint8
            "video.wrist": wrist_img,
            "state.single_arm": INIT_STATE[..., :9],      # np.float32
            "state.gripper": INIT_STATE[..., 9:],         # np.float32
            "annotation.human.task_description": [LANG_INSTRUCTION]
        }


        with torch.no_grad():
            output = policy.get_action(step_data)

            # Preserve full shape: (16, 10)
            single_arm = torch.from_numpy(output["action.single_arm"])    # (16, 9)
            gripper = torch.from_numpy(output["action.gripper"])          # (16, 1)
            action = torch.cat([single_arm, gripper], dim=-1).cpu().numpy()  # (16, 10)

        actions.append(action)
        print(f"[{i:04d}] Action:\n{action}")  # Nicely formatted, no e-N

    # Stack and save all predictions
    actions = np.stack(actions, axis=0)  # (N, 16, 10)
    np.save("predicted_actions.npy", actions)
    print("\nAll actions saved to predicted_actions.npy")