import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder = "/home/sam/xArm-MSL-ROS/msl_scripts/trajectory_plot/data_sim/pusht"

# --- Load rollout.pkl ---
pkl_path = os.path.join(folder, "rollout.pkl")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"No rollout.pkl found at {pkl_path}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Pad to 3D
def extract_xyz(arr):
    if arr.shape[1] >= 3:
        return arr[:, :3]
    else:
        padded = np.zeros((arr.shape[0], 3))
        padded[:, :arr.shape[1]] = arr
        return padded

# Plot all episodes
for ep_idx, (ep_key, ep) in enumerate(sorted(data.items())):
    if 'rollout_actions' not in ep:
        continue

    pred = ep['rollout_actions']

    if not isinstance(pred, np.ndarray) or pred.ndim != 3:
        print(f"Skipping {ep_key}: expected shape (T, 8, 10)")
        continue

    T, D, F = pred.shape
    print(f"Episode {ep_key}: rollout shape = {pred.shape}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for d in range(D):
        action_d = pred[:, d, :]  # shape (T, 10)
        traj_d = extract_xyz(action_d)  # (T, 3)
        ax.plot(traj_d[:, 0], traj_d[:, 1], traj_d[:, 2], label=f"Action dim {d}", alpha=0.8)

        # Optionally mark start and end
        ax.scatter(traj_d[0, 0], traj_d[0, 1], traj_d[0, 2], c='black', marker='o', s=80)
        ax.scatter(traj_d[-1, 0], traj_d[-1, 1], traj_d[-1, 2], c='black', marker='X', s=80)

    ax.set_title(f"Predicted Action Trajectories - {ep_key}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()