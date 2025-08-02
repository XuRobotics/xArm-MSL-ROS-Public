import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import pandas as pd
import matplotlib.cm as cm
import random
import glob

from matplotlib import colormaps

# === CONFIG ===
# Path where individual episode action files are saved
CKPT_DIR = "/home/xarm/bags/msl_bags/act_checkpoints/absolute_action_run4"  # change as needed
output_dir = "/home/xarm/xArm-MSL-ROS/msl_scripts/model_inference/3d_plot_outputs_act"
os.makedirs(output_dir, exist_ok=True)

folder_path_main = "/home/xarm/xArm-MSL-ROS/msl_scripts/model_inference/data_traj"

plot_training_trajs = True

# Collect all predicted action .npy files

all_actions = []
max_len = 0

action_files = sorted(glob.glob(os.path.join(CKPT_DIR, "predicted_actions_episode_*.npy")))
print(f"Found {len(action_files)} action files in {CKPT_DIR}")

# First pass: find max length
for fpath in action_files:
    arr = np.load(fpath)
    max_len = max(max_len, arr.shape[0])

# Second pass: load and pad
for fpath in action_files:
    arr = np.load(fpath)
    pad_len = max_len - arr.shape[0]
    if pad_len > 0:
        pad = np.full((pad_len, 10), np.nan, dtype=np.float32)  # or repeat last action
        arr = np.concatenate([arr, pad], axis=0)
    all_actions.append(arr)

merged_actions = np.stack(all_actions, axis=0)  # shape: (num_episodes, max_len, 10)
OUTPUT_FILE = os.path.join("predicted_actions_ACT.npy")
np.save(OUTPUT_FILE, merged_actions)
print(f"Saved: {merged_actions.shape}")

input_npy_path = OUTPUT_FILE


# === LOAD ROLLOUT DATA ===
actions = np.load(input_npy_path)  # shape: (num_episodes, max_timesteps, 10)
print(f"Loaded actions with shape: {actions.shape}")

# Extract end-effector positions: [:, :, 0:3]
positions = actions[:, :, 0:3]  # shape: (num_episodes, max_timesteps, 3)
print(f"Extracted positions with shape: {positions.shape}")

# === LOAD TRAINING TRAJECTORIES ===
files_main = sorted(
    [f for f in os.listdir(folder_path_main) if re.match(r"traj_\d+\.csv$", f)],
    key=lambda x: int(re.findall(r'\d+', x)[0])
)

# === RANDOMIZED COLOR MAP ===
num_trajs = min(len(files_main), positions.shape[0])
print(f"Number of trajectories to plot: {num_trajs}")

# Choose a wide colormap for variety
cmap = cm.get_cmap('hsv', num_trajs * 2)

# Generate a list of random colors
color_palette = [cmap(i) for i in random.sample(range(cmap.N), num_trajs)]

# === PLOT ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=45)

alpha_val = 0.85  # consistent, less transparent

# Plot rollout trajectories
for i in range(num_trajs):
    color = color_palette[i]
    x, y, z = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]
    ax.plot(x, y, z, alpha=alpha_val, linewidth=2, color=color, label=f'Rollout {i+1}')
    print(f"rollout traj length: {len(x)}")
    # input("Press Enter to continue...")  # Debugging pause
    ax.scatter(x[-1], y[-1], z[-1], color=color, s=50)


training_show_percentage = 0.5
training_subsample_step_size = 10
if plot_training_trajs:
    # Plot training trajectories
    for i in range(num_trajs):
        file_path = os.path.join(folder_path_main, files_main[i])
        data = pd.read_csv(file_path, header=None)
        if data.shape[1] < 3:
            continue

        x = data.iloc[:, 0]
        y = data.iloc[:, 1]
        z = data.iloc[:, 2]
        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            continue
        training_show_horizon = int(len(x) * training_show_percentage)
        x = x.iloc[:training_show_horizon:training_subsample_step_size]
        y = y.iloc[:training_show_horizon:training_subsample_step_size]
        z = z.iloc[:training_show_horizon:training_subsample_step_size]

        color = color_palette[i]
        ax.plot(x, y, z, alpha=0.6, linewidth=1, linestyle='--', color=color, label=f'Train {i+1}')
        # ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=color, s=50)

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
