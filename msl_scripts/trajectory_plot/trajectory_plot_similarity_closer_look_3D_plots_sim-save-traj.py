import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder = "/home/sam/xArm-MSL-ROS/msl_scripts/trajectory_plot/data_sim/square"
pkl_path = os.path.join(folder, "rollout.pkl")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"No rollout.pkl found at {pkl_path}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

train_data = data.get("train", {})
eval_data = data.get("eval", {})

def extract_xyz(arr):
    if arr.ndim != 2:
        raise ValueError("Array must be 2D")
    if arr.shape[1] >= 3:
        return arr[:, :3]
    else:
        padded = np.zeros((arr.shape[0], 3))
        padded[:, :arr.shape[1]] = arr
        return padded


save_base_dir = os.path.join(folder, "sim_output_traj")
os.makedirs(save_base_dir, exist_ok=True)
os.makedirs(os.path.join(save_base_dir, "train_gt"), exist_ok=True)
os.makedirs(os.path.join(save_base_dir, "eval_gt"), exist_ok=True)
os.makedirs(os.path.join(save_base_dir, "eval_pred"), exist_ok=True)

# Save training ground truth trajectories
for idx, (ep_key, ep) in enumerate(train_data.items()):
    if 'training_actions' not in ep:
        continue
    train = ep['training_actions']
    if not isinstance(train, np.ndarray) or train.ndim != 2:
        continue
    train_xyz = extract_xyz(train)
    out_path = os.path.join(save_base_dir, "train_gt", f"traj_{idx}.csv")
    np.savetxt(out_path, train_xyz, delimiter=",")
    print(f"Saved train GT: {out_path}")
# Save eval ground truth and stitched prediction trajectories (skip first/last 10, step 8)
for idx, (ep_key, ep) in enumerate(eval_data.items()):
    if 'rollout_actions' not in ep or 'training_actions' not in ep:
        continue

    rollout = ep['rollout_actions']
    gt = ep['training_actions']

    if not (isinstance(rollout, np.ndarray) and rollout.ndim == 3 and rollout.shape[1] == 8):
        continue
    if not (isinstance(gt, np.ndarray) and gt.ndim == 2):
        continue

    # Save eval ground truth
    gt_xyz = extract_xyz(gt)
    eval_gt_path = os.path.join(save_base_dir, "eval_gt", f"traj_{idx}.csv")
    np.savetxt(eval_gt_path, gt_xyz, delimiter=",")
    print(f"Saved eval GT: {eval_gt_path}")

    # Compute stitched prediction: skip first 10 and last 10, step size 8
    start_idx, end_idx = 10, rollout.shape[0] - 10
    if end_idx <= start_idx:
        print(f"Skipped eval_pred for idx {idx}: not enough timesteps after slicing.")
        continue

    stitched_pred = extract_xyz(rollout[start_idx:end_idx:8, 0, :])
    eval_pred_path = os.path.join(save_base_dir, "eval_pred", f"traj_{idx}.csv")
    np.savetxt(eval_pred_path, stitched_pred, delimiter=",")
    print(f"Saved stitched eval prediction (step=8, skip head/tail 10): {eval_pred_path}")
