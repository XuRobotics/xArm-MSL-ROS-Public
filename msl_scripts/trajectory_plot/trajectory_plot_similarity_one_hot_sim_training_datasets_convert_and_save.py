import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder = "/home/sam/xArm-MSL-ROS/msl_scripts/trajectory_plot/data_sim/can"
pkl_path = os.path.join(folder, "rollout.pkl")
# --- Save training and rollout trajectories ---
output_folder = "GT_and_rollout_trajectories_can"

if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"No rollout.pkl found at {pkl_path}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

train_data = data.get("train", {})

# train_data = data.get("eval", {})

def extract_xyz(arr):
    """Extract the first 3 dimensions (X, Y, Z) from a 2D array."""
    if arr.ndim != 2:
        raise ValueError("Array must be 2D")
    if arr.shape[1] >= 3:
        return arr[:, :3]
    else:
        padded = np.zeros((arr.shape[0], 3))
        padded[:, :arr.shape[1]] = arr
        return padded

os.makedirs(output_folder, exist_ok=True)

# --- Plot and save all training trajectories (ground truth and rollouts) ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for ep_key, ep in train_data.items():
    if 'training_actions' not in ep or 'rollout_actions' not in ep:
        continue
    
    # Extract the ground truth (training actions)
    train = ep['training_actions']
    if not isinstance(train, np.ndarray) or train.ndim != 2:
        continue
    train_xyz = extract_xyz(train)
    
    # Save the training (ground truth) trajectory to CSV file
    train_filename = f"{output_folder}/train_trajectory_{ep_key}.csv"
    np.savetxt(train_filename, train_xyz, delimiter=",", header="X,Y,Z", comments="")
    print(f"Saved training trajectory: {train_filename}")
    
    # Plot the training (ground truth) trajectory in green
    ax.plot(train_xyz[:, 0], train_xyz[:, 1], train_xyz[:, 2],
            color='green', alpha=0.7, linewidth=1, label=f"Train Ground Truth {ep_key}")

    # Extract the rollout (predictions) trajectory
    rollout = ep['rollout_actions']
    if not isinstance(rollout, np.ndarray) or rollout.ndim != 3:
        continue
    
    # Ensure we are only extracting the XYZ components from the rollout
    # Select every 'horizon'-th step from the rollout
    horizon = 8  # Assuming the horizon is 8 as per the original code
    rollout_xyz = rollout[::horizon, :, :]  # Only select every horizon-th step

    # Extract XYZ for the selected rollout actions
    rollout_xyz = np.array([extract_xyz(rollout_xyz[t, :, :]) for t in range(rollout_xyz.shape[0])])
    
    # rollout_xyz = np.array([extract_xyz(rollout[t, :, :]) for t in range(rollout.shape[0])])

    # Flatten the rollout trajectory into a 2D array of XYZ values
    rollout_xyz_flat = rollout_xyz.reshape(-1, 3)

    # Save the rollout trajectory (predicted data) to CSV file
    rollout_filename = f"{output_folder}/rollout_trajectory_{ep_key}.csv"
    np.savetxt(rollout_filename, rollout_xyz_flat, delimiter=",", header="X,Y,Z", comments="")

    # Plot the rollout trajectory in purple
    ax.plot(rollout_xyz_flat[:, 0], rollout_xyz_flat[:, 1], rollout_xyz_flat[:, 2],
            color='purple', alpha=0.6, linewidth=2, label=f"Rollout {ep_key}")

# --- Labels and legend ---
ax.set_title("Training Trajectories: Ground Truth and Rollout")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Handle the legend and avoid duplicates by setting it only once
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:10], labels[:10])  # Adjust the number for your legend labels as per your need

plt.tight_layout()
plt.show()
