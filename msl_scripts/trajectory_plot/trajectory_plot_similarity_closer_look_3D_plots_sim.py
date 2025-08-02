import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



use_evaluation_data = True  # Set to True to use evaluation data, False for training data
folder = "/home/sam/xArm-MSL-ROS/msl_scripts/trajectory_plot/data_sim/square"
pkl_path = os.path.join(folder, "rollout.pkl")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"No rollout.pkl found at {pkl_path}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
    
# print out the keys in data
print("Keys in data:")
for key in data.keys():
    print(f" - {key}")
    
if use_evaluation_data:
    # only take out the data with 'eval'
    data = data.get('eval', {})
else:
    # only take out the data with 'train'
    data = data.get('train', {})

def extract_xyz(arr):
    if arr.ndim != 2:
        raise ValueError("Array must be 2D")
    if arr.shape[1] >= 3:
        return arr[:, :3]
    else:
        padded = np.zeros((arr.shape[0], 3))
        padded[:, :arr.shape[1]] = arr
        return padded

episode_items = sorted(data.items())
batch_size = 1

for batch_start in range(0, len(episode_items), batch_size):
    batch = episode_items[batch_start:batch_start + batch_size]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for ep_key, ep in batch:
        if 'training_actions' not in ep or 'rollout_actions' not in ep:
            continue

        train = ep['training_actions']
        rollout = ep['rollout_actions']

        if not (isinstance(rollout, np.ndarray) and rollout.ndim == 3 and rollout.shape[1] == 8):
            print(f"ERROR: rollout_actions in {ep_key} is not of shape (T, 8, D)")
            continue

        if not isinstance(train, np.ndarray) or train.ndim != 2:
            continue

        skip_interval = 8 # simliar to your rollout horizon so that the trajectory stitch together
        train = train[skip_interval:]  # Optional alignment
        train_xyz = extract_xyz(train)
        T = min(len(train_xyz), rollout.shape[0])

        # Plot ground truth
        ax.plot(train_xyz[:T, 0], train_xyz[:T, 1], train_xyz[:T, 2],
                color='green', alpha=0.9, linewidth=2, label='Training' if batch_start == 0 else None)

        # Color map: orange to red
        cmap = plt.cm.autumn  # Orange → Red
        colors = [cmap(i / T) for i in range(T)]

        for t in range(0, T-skip_interval, skip_interval):
            # print(f"Plotting rollout for episode {ep_key}, time step {t}/{T}")
            action_set = rollout[t, :, :]
            action_xyz = extract_xyz(action_set)
            ax.plot(action_xyz[:, 0], action_xyz[:, 1], action_xyz[:, 2],
                    color=colors[t], alpha=0.8, linewidth=3)

        # Start & goal markers from rollout[0]
        first_pred = extract_xyz(rollout[:, 0, :])
        ax.scatter(first_pred[0, 0], first_pred[0, 1], first_pred[0, 2], c='black', marker='o', s=40)
        ax.scatter(first_pred[T-1, 0], first_pred[T-1, 1], first_pred[T-1, 2], c='black', marker='X', s=40)

    ax.set_title(f"Episodes {batch_start}–{batch_start + len(batch) - 1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Legend only once
    green_line = plt.Line2D([0], [0], color='green', label='Training')
    gradient_line = plt.Line2D([0], [0], color='orangered', label='Rollout Arcs (T=0→T=N)')
    black_o = plt.Line2D([0], [0], marker='o', color='black', linestyle='None', label='Start')
    black_x = plt.Line2D([0], [0], marker='X', color='black', linestyle='None', label='Goal')
    ax.legend(handles=[green_line, gradient_line, black_o, black_x])

    plt.tight_layout()
    plt.show()
