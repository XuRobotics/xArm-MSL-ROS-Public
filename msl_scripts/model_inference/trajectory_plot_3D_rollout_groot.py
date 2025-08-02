import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import pandas as pd
import matplotlib.cm as cm
import random

# === CONFIG ===
input_npy_path = "/home/xarm/xArm-MSL-ROS/msl_scripts/model_inference/predicted_actions.npy"
output_dir = "/home/xarm/xArm-MSL-ROS/msl_scripts/model_inference/3d_plot_outputs"
os.makedirs(output_dir, exist_ok=True)

training_show_horizon_percent = 1.0
train_subsample_step_size = 5  # for training trajectories, to reduce number of points plotted

inf_subsample_step_size = 1 # for inference trajectories, to plot all points
num_inf_traj_to_show = 30
num_train_traj_to_show = 30

folder_path_main = "/home/xarm/xArm-MSL-ROS/msl_scripts/model_inference/data_traj_OWR"
print(f"Using folder path for training trajectories: {folder_path_main}")
input(f"CONFIRM THAT THIS IS THE CORRECT FOLDER PATH FOR TRAINING TRAJECTORIES, press Enter to continue or Ctrl+C to exit...")

# === LOAD ROLLOUT DATA ===
actions = np.load(input_npy_path)  # shape: (30, 1, 16, 10)
print(f"Loaded actions with shape: {actions.shape}")

positions = actions[:, 0, :, 0:3]  # shape: (30, 16, 3)
print(f"Extracted positions with shape: {positions.shape}")

# === LOAD TRAINING TRAJECTORIES ===
files_main = sorted(
    [f for f in os.listdir(folder_path_main) if re.match(r"traj_\d+\.csv$", f)],
    key=lambda x: int(re.findall(r'\d+', x)[0])
)

# === RANDOMIZED COLOR MAP ===
num_trajs = min(len(files_main), positions.shape[0])
colormap = plt.colormaps['gist_ncar']
colors = [colormap(i / (num_trajs * 2)) for i in range(num_trajs * 2)]
indices = list(range(num_trajs * 2))
random.shuffle(indices)
color_indices = indices[:num_trajs]

# === PLOT ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=45)

alpha_val = 0.85  # consistent, less transparent

# Plot rollout trajectories
for i in range(num_trajs):
    if i >= num_inf_traj_to_show:
        break  # Limit to num_inf_traj_to_show
    color = colors[color_indices[i]]
    # x, y, z = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]
    # subsample use inf_subsample_step_size
    x = positions[i, ::inf_subsample_step_size, 0]
    y = positions[i, ::inf_subsample_step_size, 1]
    z = positions[i, ::inf_subsample_step_size, 2]
    ax.plot(x, y, z, alpha=alpha_val, linewidth=2, color=color, label=f'Rollout {i+1}')
    # ax.scatter(x[-1], y[-1], z[-1], color=color, s=50)

# Plot training trajectories
for i in range(num_trajs):
    if i >= num_train_traj_to_show:
        break  # Limit to num_train_traj_to_show
    file_path = os.path.join(folder_path_main, files_main[i])
    data = pd.read_csv(file_path, header=None)
    if data.shape[1] < 3:
        continue

    # x = data.iloc[:training_show_horizon, 0]
    # y = data.iloc[:training_show_horizon, 1]
    # z = data.iloc[:training_show_horizon, 2]
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        continue  # Skip empty trajectories
    training_show_horizon = int(len(x) * training_show_horizon_percent)
    x = x[:training_show_horizon]
    y = y[:training_show_horizon]
    z = z[:training_show_horizon]

    color = colors[color_indices[i]]
    ax.plot(x[::train_subsample_step_size],
             y[::train_subsample_step_size],
             z[::train_subsample_step_size],
             alpha=0.6, linewidth=2, linestyle='--', color='gray', label=f'Train {i+1}')
    # ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color='gray', s=50)

# Labels and title
ax.set_xlabel("X (mm)", fontsize=12)
ax.set_ylabel("Y (mm)", fontsize=12)
ax.set_zlabel("Z (mm)", fontsize=12)
ax.set_title("Rollout and Training 3D Trajectories", fontsize=14)
ax.set_facecolor('whitesmoke')

# Combine legend entries for clarity
handles, labels = ax.get_legend_handles_labels()
unique = dict()
for h, l in zip(handles, labels):
    if l not in unique:
        unique[l] = h
ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=9, ncol=2)

plt.tight_layout()
plt.show()
