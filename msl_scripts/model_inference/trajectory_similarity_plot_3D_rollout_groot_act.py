import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial import cKDTree
import argparse
import shutil

# === CONFIG ===
# training_folder = "data_traj"
training_folder = 'data_traj_OWR'
print(f"Using folder path for training trajectories: {training_folder}")
input(f"CONFIRM THAT THIS IS THE CORRECT FOLDER PATH FOR TRAINING TRAJECTORIES, press Enter to continue or Ctrl+C to exit...")
# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="Compute rollout-to-training trajectory similarity.")
parser.add_argument('--for_act', type=lambda x: x.lower() == 'true', default=False,
                    help='Use ACT rollouts instead of standard GR00T outputs. Accepts True or False.')
args = parser.parse_args()
for_act = args.for_act

if not for_act:
    rollout_npy_path = "predicted_actions.npy"
    training_traj_end_stamp = -1
    output_folder = "similarity_plots_groot"
    max_dist_for_similarity = 15

    max_dist_for_raw_plot = 150
else:
    training_traj_end_stamp = -1
    rollout_npy_path = "predicted_actions_ACT.npy"
    output_folder = "similarity_plots_act"
    max_dist_for_raw_plot = 150
    max_dist_for_similarity = 15
num_trajs_to_plot = 100 

# remove output_folder if it exists
if os.path.exists(output_folder):
    print(f"Removing existing output folder: {output_folder}")
    shutil.rmtree(output_folder)


os.makedirs(output_folder, exist_ok=True)

# === LOAD TRAINING TRAJS ===
def load_trajectories(folder):
    traj_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".csv")],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    trajectories = []
    print(f"Loading training data from {folder}")
    for file in traj_files:
        data = pd.read_csv(os.path.join(folder, file), header=None)
        if data.shape[1] < 3:
            print(f"Skipping {file}, only {data.shape[1]} columns")
            continue
        trajectories.append(data.iloc[:training_traj_end_stamp, :3].values)
    print(f"Loaded {len(trajectories)} trajectories from {folder}")
    return trajectories

# === LOAD ROLLOUT TRAJS FROM NPY ===
def load_rollout_from_npy(npy_path):
    actions = np.load(npy_path)
    print(f"Loaded rollout actions from {npy_path} with shape: {actions.shape}")
    
    if actions.ndim == 4:
        rollouts = [actions[i, 0, :, 0:3] for i in range(actions.shape[0])]
    elif actions.ndim == 3:
        rollouts = [actions[i, :, 0:3] for i in range(actions.shape[0])]
    else:
        raise ValueError(f"Unexpected actions shape: {actions.shape}")
    
    cleaned_rollouts = []

    # Filter out fully padded (all-NaN) rollouts
    for i, traj in enumerate(rollouts):
        nan_mask = np.isnan(traj).any(axis=1)
        nan_indices = np.where(nan_mask)[0]

        if len(nan_indices) > 0:
            print(f"Rollout {i}: removed {len(nan_indices)} NaN rows at indices {nan_indices.tolist()}")

        cleaned_traj = traj[~nan_mask]
        cleaned_rollouts.append(cleaned_traj)

    print(f"Cleaned {len(cleaned_rollouts)} rollout trajectories (NaN rows removed)")
    return cleaned_rollouts

# === POINTWISE DISTANCE ===
def compute_pointwise_distance(rollout, train):
    tree = cKDTree(train)
    dists, _ = tree.query(rollout, k=1)
    return dists.tolist()

# === MAIN ===
training_trajs = load_trajectories(training_folder)
rollout_trajs = load_rollout_from_npy(rollout_npy_path)
if len(rollout_trajs) > num_trajs_to_plot:
    rollout_trajs = rollout_trajs[:num_trajs_to_plot]

n_rollout = len(rollout_trajs)
n_train = len(training_trajs)

# === SAVE INDIVIDUAL PLOTS ===
for idx, rollout_traj in enumerate(rollout_trajs):
    fig, ax = plt.subplots(figsize=(10, 5))
    similarity_scores = []
    distances_cropped = []

    for train_traj in training_trajs:
        distances = compute_pointwise_distance(rollout_traj, train_traj)
        sorted_distances = sorted(distances)
        cutoff = int(len(sorted_distances) * 0.95)
        mean_distance = np.mean(sorted_distances[:cutoff])
        sim_score = max(0, 1 - mean_distance / max_dist_for_similarity)

        similarity_scores.append(sim_score)
        distances_cropped.append(min(mean_distance, max_dist_for_raw_plot))

    # Bar plot
    bars = ax.bar(range(n_train), similarity_scores, color=f'C{idx % 10}', alpha=0.9)
    best_idx = np.argmax(similarity_scores)
    ax.annotate(f'{distances_cropped[best_idx]:.1f} mm',
                xy=(best_idx, similarity_scores[best_idx]),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=8)
    ax.set_title(f"Rollout Traj #{idx}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training Trajectory Index")
    ax.set_ylabel("Similarity Score")

    # Distance curve
    ax2 = ax.twinx()
    x_vals = np.arange(n_train)
    ax2.plot(x_vals, distances_cropped, color='blue', alpha=0.5, linewidth=1.2)
    # Red dot for lowest distance
    min_dist_idx = np.argmin(distances_cropped)
    ax2.plot(min_dist_idx, distances_cropped[min_dist_idx], 'ro')
    ax2.annotate(f"Min Dist\nTrain #{min_dist_idx}",
                 xy=(min_dist_idx, distances_cropped[min_dist_idx]),
                 xytext=(10, -20), textcoords='offset points',
                 fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
    ax2.set_ylim(0, max_dist_for_raw_plot)
    ax2.set_ylabel("Distance (mm)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"traj_{idx:03d}.png"))
    plt.close()

print(f"Saved {n_rollout} plots to '{output_folder}' folder.")

# === FULL COMPOSITE GRID PLOT ===
n_cols = 2
n_rows = int(np.ceil(n_rollout / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for idx, (rollout_traj, ax) in enumerate(zip(rollout_trajs, axes)):
    similarity_scores = []
    distances_cropped = []

    for train_traj in training_trajs:
        distances = compute_pointwise_distance(rollout_traj, train_traj)
        sorted_distances = sorted(distances)
        cutoff = int(len(sorted_distances) * 0.95)
        mean_distance = np.mean(sorted_distances[:cutoff])
        sim_score = max(0, 1 - mean_distance / max_dist_for_similarity)
        similarity_scores.append(sim_score)
        distances_cropped.append(min(mean_distance, max_dist_for_raw_plot))

    best_idx = np.argmax(similarity_scores)
    min_dist_idx = np.argmin(distances_cropped)

    # Bar plot
    ax.bar(range(n_train), similarity_scores, color=f'C{idx % 10}', alpha=0.9)
    ax.annotate(f'{distances_cropped[best_idx]:.1f} mm',
                xy=(best_idx, similarity_scores[best_idx]),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=8)
    ax.set_title(f"Rollout Traj #{idx}", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training Trajectory Index")
    ax.set_ylabel("Similarity Score")

    # Distance curve
    ax2 = ax.twinx()
    x_vals = np.arange(n_train)
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), n_train * 2)
    spline = make_interp_spline(x_vals, distances_cropped, k=1)
    y_smooth = spline(x_smooth)
    ax2.plot(x_smooth, y_smooth, color='blue', alpha=0.5, linewidth=1.2)

    ax2.plot(min_dist_idx, distances_cropped[min_dist_idx], 'ro')
    ax2.annotate(f"Min Dist\nTrain #{min_dist_idx}",
                 xy=(min_dist_idx, distances_cropped[min_dist_idx]),
                 xytext=(10, -20), textcoords='offset points',
                 fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
    ax2.set_ylim(0, max_dist_for_raw_plot)
    ax2.set_ylabel("Distance (mm)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

# Hide unused subplots
for ax in axes[n_rollout:]:
    ax.axis('off')

fig.suptitle("Similarity Scores (bar) and Mean Distances (line) of Rollout vs All Training", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.text(0.5, 0.01, "Training Trajectory Index", ha='center')
# save this plot as well
plt.savefig(os.path.join(output_folder, "similarity_composite_plot.png"))
plt.show()

