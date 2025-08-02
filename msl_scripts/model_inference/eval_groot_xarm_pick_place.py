import os
import numpy as np
import cv2
import torch
import json
from tqdm import tqdm
import re
import sys
sys.path.append("/home/xarm/Isaac-GR00T")


import time

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP


# === CONFIG ===
root_dir = "/home/xarm/bags/msl_bags/OWR-bags-7-28-2025-robot-demo-for-inference"
EXTRACTED_ROOT = os.path.join(root_dir, "extracted_images_and_poses_OWR")
MODALITY_JSON = "/home/xarm/bags/msl_bags/converted_groot_data_absolute_OWR/meta/modality.json"
CHECKPOINT_PATH = "/home/xarm/bags/msl_bags/groot_checkpoints/xarm_OWR_absolute_pose_batch_16_horizon_200/checkpoint-73000"
LANG_INSTRUCTION = "Pick and place the object."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# set printing to be 2 decimal places
np.set_printoptions(precision=3, suppress=True, linewidth=200, threshold=10000)


# Only for sanity check
INIT_STATE_VECTOR = [
    -0.9996057, 0.00008499, 0.00018754,
    0.9999933, 0.02808065, -0.00365329,
    259.1312, 2.76629, 258.1401,
    0.9282353
]

def load_modality_config(path):
    with open(path, "r") as f:
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


if __name__ == "__main__":
    log_file_path = "./inference_times_groot.txt"
    log_file = open(log_file_path, "w")
    log_file.write("=== Gr00t Inference Timing Log ===\n")

    modality_config = load_modality_config(MODALITY_JSON)
    # modality_config["action"].delta_indices = list(range(100))
    if modality_config["action"].delta_indices != list(range(200)):
        print(f"Expected action delta_indices to be [0, 1, ..., 199] for 200-step horizon, but got {modality_config['action'].delta_indices}")
        print(f"Expected action delta_indices to be [0, 1, ..., 199] for 200-step horizon, but got {modality_config['action'].delta_indices}")
        print(f"Expected action delta_indices to be [0, 1, ..., 199] for 200-step horizon, but got {modality_config['action'].delta_indices}")
        # set it to [0, 1, ..., 199]
        modality_config["action"].delta_indices = list(range(200))

        print(f"Setting action delta_indices to [0, 1, ..., 199] for 200-step horizon")
        input("Confirm that you want action horizon to be 200 steps, press Enter to continue or Ctrl+C to exit...")
    
    transform_config = DATA_CONFIG_MAP["xarm_dualcam"].transform()

    policy = Gr00tPolicy(
        model_path=CHECKPOINT_PATH,
        modality_config=modality_config,
        modality_transform=transform_config,
        embodiment_tag=EmbodimentTag.OXE_DROID,
        device=DEVICE
    )
    policy.model.to(DEVICE)
    policy.model.action_head.to(DEVICE)

    def extract_timestamp(folder_name):
        # Extract the timestamp part: "2025-03-19-10-52-50" from folder name like "xarm_demo_2025-03-19-10-52-50"
        match = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", folder_name)
        return match.group() if match else ""

    folders = [
        f for f in os.listdir(EXTRACTED_ROOT)
        if os.path.isdir(os.path.join(EXTRACTED_ROOT, f))
    ]
    folders = sorted(folders, key=extract_timestamp)


    all_actions = []

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"WARNING: state_vec is modified from [position (mm), orientation (6D), gripper (0-850)] to match model expectations, which expects [orientation (6D), position (mm), gripper (0-1)]")
    
    inference_times = []

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"WARNING: state_vec is modified from [position (mm), orientation (6D), gripper (0-850)] to match model expectations, which expects [orientation (6D), position (mm), gripper (0-1)]")

    for folder in tqdm(folders, desc="Running inference"):
        folder_path = os.path.join(EXTRACTED_ROOT, folder)
        state_path = os.path.join(folder_path, "state.npy")
        front_path = os.path.join(folder_path, "camera_0_frame_0000.png")
        wrist_path = os.path.join(folder_path, "camera_1_frame_0000.png")

        print(f"Paths: {state_path}, {front_path}, {wrist_path}")

        if not os.path.exists(state_path) or not os.path.exists(front_path) or not os.path.exists(wrist_path):
            print(f"[{folder}] Missing data, skipping.")
            continue

        state_vec_ori = np.load(state_path)  # shape (10,)

        state_vec = np.zeros_like(state_vec_ori)
        state_vec[:6] = state_vec_ori[3:9]  # orientation (6D)
        state_vec[6:9] = state_vec_ori[:3]  # position (mm)
        state_vec[9] = state_vec_ori[9] / 850.0  # gripper (0–1)

        diff = np.abs(state_vec - INIT_STATE_VECTOR)
        print(f"State vector diff: {diff}")
        print(f"current diff between state vector and initial state in rotation (6D), position (mm), gripper (0-1): {diff[:6]}, {diff[6:9]}, {diff[9]}")
        if np.any(diff[:6] > 0.2) or np.any(diff[6:9] > 20) or diff[9] > 0.2:
            print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Warning: [{folder}] State vector diff is too large from the expected initial state.")
            print(f"IF you are doing a different task than Distribution Pick and Place, this is OK.")
            print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        print(f"make sure that the model is expecting gripper of 0-1, not 0-850")

        # Front image shape: (720, 1280, 3), Wrist image shape: (720, 1280, 3)
        front_img = cv2.cvtColor(cv2.imread(front_path), cv2.COLOR_BGR2RGB)
        wrist_img = cv2.cvtColor(cv2.imread(wrist_path), cv2.COLOR_BGR2RGB)


        front_input = front_img[np.newaxis, np.newaxis, ...]
        wrist_input = wrist_img[np.newaxis, np.newaxis, ...]
        INIT_STATE = np.array([[state_vec]], dtype=np.float32)

        step_data = {
            "video.front": front_input,
            "video.wrist": wrist_input,
            "state.single_arm": INIT_STATE[..., :9],
            "state.gripper": INIT_STATE[..., 9:],
            "annotation.human.task_description": [LANG_INSTRUCTION]
        }

        start_time = time.time()
        with torch.no_grad():
            output = policy.get_action(step_data)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        log_file.write(f"[{folder}] Inference time: {inference_time * 1000:.2f} ms\n")


        single_arm = torch.from_numpy(output["action.single_arm"])
        gripper = torch.from_numpy(output["action.gripper"])
        action = torch.cat([single_arm, gripper], dim=-1).cpu().numpy()

        all_actions.append(action)
        print(f"[{folder}] Inference complete - time: {inference_time * 1000:.2f} ms")

    if inference_times:
        avg_ms = np.mean(inference_times) * 1000
        print(f"\n=== Average inference time: {avg_ms:.2f} ms ({1000 / avg_ms:.2f} Hz) ===")
        # get the median inference time
        median_ms = np.median(inference_times) * 1000
        print(f"=== Median inference time: {median_ms:.2f} ms ({1000 / median_ms:.2f} Hz) ===")
        log_file.write(f"\nAverage inference time: {avg_ms:.2f} ms ({1000 / avg_ms:.2f} Hz)\n")
        log_file.write(f"Median inference time: {median_ms:.2f} ms ({1000 / median_ms:.2f} Hz)\n")
    log_file.close()
    print(f"Inference log saved to: {log_file_path}")


    # Stack all actions and save
    if all_actions:
        all_actions = np.stack(all_actions, axis=0)  # Shape: (N, 16, 10)
        output_path = "predicted_actions.npy"
        np.save(output_path, all_actions)
        print(f"\nAll predicted actions saved to: {output_path}")
    else:
        print("No valid predictions were made.")
