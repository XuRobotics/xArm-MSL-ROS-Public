import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder = "/home/sam/xArm-MSL-ROS/msl_scripts/trajectory_plot/data_sim/can"
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

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# --- Plot all training trajectories ---
for ep_key, ep in train_data.items():
    if 'training_actions' not in ep:
        continue
    train = ep['training_actions']
    if not isinstance(train, np.ndarray) or train.ndim != 2:
        continue
    train_xyz = extract_xyz(train)
    ax.plot(train_xyz[:, 0], train_xyz[:, 1], train_xyz[:, 2],
            color='green', alpha=0.7, linewidth=1)

# --- Overlay eval rollouts + ground truth ---
colors = ['purple', 'purple', 'purple', 'purple', 'purple', 'purple', 'purple', 'purple']
eval_eps = list(eval_data.items())

for idx, (ep_key, ep) in enumerate(eval_eps):
    if 'rollout_actions' not in ep or 'training_actions' not in ep:
        continue

    rollout = ep['rollout_actions']
    gt = ep['training_actions']

    if not (isinstance(rollout, np.ndarray) and rollout.ndim == 3 and rollout.shape[1] == 8):
        continue
    if not (isinstance(gt, np.ndarray) and gt.ndim == 2):
        continue

    color = colors[idx] if idx < len(colors) else 'gray'
    T = rollout.shape[0]

    # --- Plot rollout arcs ---
    for t in range(0, T - 10, 7):
        action_set = rollout[t, :, :]
        action_xyz = extract_xyz(action_set)
        ax.plot(action_xyz[:, 0], action_xyz[:, 1], action_xyz[:, 2],
                color=color, alpha=0.6, linewidth=4)

    # --- Plot eval ground truth trajectory in red ---
    gt_xyz = extract_xyz(gt)
    
    # dotted line for ground truth
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
            color='red', linestyle='--', linewidth=2, label='Eval Ground Truth')

    # --- Start and goal markers from rollout ---
    first_pred = extract_xyz(rollout[:, 0, :])
    ax.scatter(first_pred[0, 0], first_pred[0, 1], first_pred[0, 2], c='black', marker='o', s=80)
    ax.scatter(first_pred[-1, 0], first_pred[-1, 1], first_pred[-1, 2], c='black', marker='X', s=80)

# --- Labels and legend ---
ax.set_title("Overlay: Eval Rollouts, Ground Truth, and Train Trajectories")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

legend_lines = [
    plt.Line2D([0], [0], color='green', label='Training (Train Set)'),
    plt.Line2D([0], [0], color='red', label='Eval Ground Truth'),
    plt.Line2D([0], [0], color='purple', label='Eval Rollout'),
    plt.Line2D([0], [0], marker='o', color='black', linestyle='None', label='Start'),
    plt.Line2D([0], [0], marker='X', color='black', linestyle='None', label='Goal'),
]
ax.legend(handles=legend_lines)

plt.tight_layout()
plt.show()
