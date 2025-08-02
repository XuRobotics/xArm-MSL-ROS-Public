import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# === TO load rollout.pkl ===
folder = "/home/sam/xArm-MSL-ROS/msl_scripts/trajectory_plot/data_sim/pusht"

# === Utility ===
def extract_xyz(array):
    """Ensure array has shape (T, 3) by slicing or padding."""
    if array.shape[1] >= 3:
        return array[:, :3]
    else:
        padded = np.zeros((array.shape[0], 3))
        padded[:, :array.shape[1]] = array
        return padded

def compute_pointwise_distance(traj1, traj2):
    """Compute min distance from each point in traj1 to traj2."""
    return [np.min(np.linalg.norm(traj2 - pt, axis=1)) for pt in traj1]

# === Config ===
max_dist = 15           # max distance for similarity score
max_plot_dist = 100     # y-axis limit for distance curve

# === Load rollout.pkl ===
pkl_path = os.path.join(folder, "rollout.pkl")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"No rollout.pkl found at {pkl_path}")

with open(pkl_path, "rb") as f:
    rollout_data = pickle.load(f)

main_trajs, target_trajs = [], []

for ep_key in sorted(rollout_data.keys()):
    ep = rollout_data[ep_key]
    if 'training_actions' not in ep or 'rollout_actions' not in ep:
        continue

    gt = ep['training_actions']
    pred = ep['rollout_actions']

    # === Extract GT ===
    if not isinstance(gt, np.ndarray) or gt.ndim != 2:
        continue
    gt_xyz = extract_xyz(gt)

    # === Extract Prediction ===
    if isinstance(pred, np.ndarray) and pred.ndim == 3:
        # (T, 8, N): use first dim, then first 3 values
        if pred.shape[2] >= 3:
            pred_decoded = pred[:, 0, :3]
        else:
            pred_decoded = pred[:, 0, :]
    elif isinstance(pred, np.ndarray) and pred.ndim == 2:
        pred_decoded = pred  # (T, 2)
    else:
        continue

    pred_xyz = extract_xyz(pred_decoded)

    # Match length
    T = min(len(gt_xyz), len(pred_xyz))
    main_trajs.append(gt_xyz[:T])
    target_trajs.append(pred_xyz[:T])

if len(main_trajs) == 0:
    raise ValueError("No valid episodes found for comparison.")

# === Plot ===
n_cols = 2
n_rows = int(np.ceil(len(target_trajs) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
axes = axes.flatten()

for idx, (gt, pred, ax) in enumerate(zip(main_trajs, target_trajs, axes)):
    distances = compute_pointwise_distance(pred, gt)
    sorted_distances = sorted(distances)
    cutoff = int(len(sorted_distances) * 0.95)
    mean_dist = np.mean(sorted_distances[:cutoff])
    similarity = max(0, 1 - mean_dist / max_dist)

    # Similarity bar
    ax.bar([0], [similarity], color='green', alpha=0.9, bottom=-0.1)
    ax.annotate(f"{mean_dist:.1f} mm", xy=(0, similarity - 0.1), xytext=(0, 1),
                textcoords='offset points', ha='center', va='bottom', fontsize=8)

    # Smoothed distance curve
    x = np.arange(len(distances))
    x_smooth = np.linspace(x.min(), x.max(), len(distances) * 2)
    spline = make_interp_spline(x, distances, k=1)
    y_smooth = spline(x_smooth)

    ax2 = ax.twinx()
    ax2.plot(x_smooth, y_smooth, color='blue', alpha=0.4)
    ax2.set_ylim(0, max_plot_dist)
    ax2.set_ylabel("Distance (mm)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax.set_ylim(0, 1)
    ax.set_title(f"Episode {idx}")
    ax.set_ylabel("Similarity Score")
    ax.set_xticks([])

# Clean unused subplots
for ax in axes[len(target_trajs):]:
    ax.axis('off')

fig.suptitle("Trajectory Similarity (Predicted vs Ground Truth)", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.show()
