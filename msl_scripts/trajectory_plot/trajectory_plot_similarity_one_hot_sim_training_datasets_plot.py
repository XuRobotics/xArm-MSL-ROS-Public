import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# Parameters
output_folder = "GT_and_rollout_trajectories_square"  # Change to your output folder
# output_folder = "GT_and_rollout_trajectories_can"  # Change to your output folder
max_train = 195
max_rollout = 195
n_rows, n_cols = 2, 2  # 13 * 15 = 195 subplots

def extract_index(filename, prefix):
    try:
        return int(filename.replace(prefix, '').split('.')[0])
    except ValueError:
        return None

# Load and sort filenames
train_files = sorted(
    [f for f in os.listdir(output_folder) if f.startswith("train_trajectory_demo_")],
    key=lambda f: extract_index(f, "train_trajectory_demo_")
)
rollout_files = sorted(
    [f for f in os.listdir(output_folder) if f.startswith("rollout_trajectory_demo_")],
    key=lambda f: extract_index(f, "rollout_trajectory_demo_")
)

# Load data
main_trajs = [pd.read_csv(os.path.join(output_folder, f), header=0).iloc[:, :3].values for f in train_files]
target_trajs = [pd.read_csv(os.path.join(output_folder, f), header=0).iloc[:, :3].values for f in rollout_files]

# Distance function
def avg_min_distance(from_traj, to_traj):
    return np.mean([
        np.min(np.linalg.norm(to_traj - pt, axis=1))
        for pt in from_traj
    ])

# Collect best similarities and distances
best_similarities = []
best_distances = []

# Plotting
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 12), sharey=True)
axes = axes.flatten()

for idx, (rollout, ax) in enumerate(zip(target_trajs[:max_rollout], axes)):
    distances_to_trains = [
        avg_min_distance(rollout, train) for train in main_trajs[:max_train]
    ]

    sorted_indices = np.argsort(distances_to_trains)
    best_idx = sorted_indices[0]
    second_best_idx = sorted_indices[1]

    best_distance = distances_to_trains[best_idx]
    best_distances.append(best_distance)

    dist_best_vs_second = avg_min_distance(main_trajs[best_idx], main_trajs[second_best_idx])
    if dist_best_vs_second == 0:
        dist_best_vs_second = 1e-6

    similarity_scores = [
        1 - (d / dist_best_vs_second) for d in distances_to_trains
    ]

    best_sim = min(1.0, max(similarity_scores))
    best_similarities.append(best_sim)

    similarity_scores_clipped = [max(s, -0.2) for s in similarity_scores]

    train_ids = list(range(len(similarity_scores)))
    ax.bar(train_ids, similarity_scores_clipped, color='green', alpha=0.8)
    ax.set_ylim(-0.2, 1.1)
    ax.set_title(f"Rollout #{idx}", fontsize=10)
    ax.set_xticks(train_ids)
    ax.set_xticklabels([f"T{i}" for i in train_ids], rotation=45)
    ax.set_ylabel("Similarity Score")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.bar(best_idx, similarity_scores_clipped[best_idx], color='orange', alpha=0.9)
    ax.text(
        best_idx,
        similarity_scores_clipped[best_idx] + 0.03,
        f"{similarity_scores_clipped[best_idx]:.2f}",
        ha='center',
        va='bottom',
        fontsize=10,
        weight='bold',
        color='black'
    )

# Hide unused subplots
for ax in axes[len(target_trajs[:max_rollout]):]:
    ax.axis("off")

# Optionally remove x-axis labels if n_rows > 5
if n_rows > 5:
    for ax in axes:
        ax.set_xticklabels([])

# Averages
average_best_sim = np.mean(best_similarities)
average_best_distance = np.mean(best_distances)
print(f"Average of highest similarity scores (one per rollout): {average_best_sim:.4f}")
print(f"Average distance to most similar ground-truth trajectory: {average_best_distance*1000:.4f}")

# Finalize plot
fig.suptitle("Similarity to Ground-Truth Trajectories (Per-Rollout Normalized)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and show plot
cur_time_str = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"trajectory_similarity_plot_{cur_time_str}.png", dpi=300)
plt.show()
