"""
MIT License

Copyright (c) 2025 Xu Liu, Chengyang He, Gadiel Sznaier Camps (with the help of 
Github Copilot) at Multi-Robot Systems Lab, Stanford University 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""


# IMPORTANT NOTE: WHY NOT USE DOT PRODUCT FOR TRAJECTORY SIMILARITY?
# Spatial proximity does NOT imply directional similarity. Dot product emphasizes directions and magnitudes, while Euclidean distance emphasizes spatial positions.
# If you're measuring how similarly two trajectories are oriented or aligned, use metrics based on angles, such as cosine similarity (normalized dot product).
# If you're measuring how close two trajectories are in spatial coordinates, use Euclidean distance.
# DOT PRODUCTS ASSUME PERFECT ALIGNMENT OF TIMESTAMPS, The dot product between two trajectories treats each trajectory as a single, flattened vector of waypoint positions. It implicitly assumes perfect alignment in timestamps and an identical number of waypoints

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

rollout_traj_prefix_is_timestamp = True # If True, should be of the format e.g.: 2025-03-31-03-36-42_traj.csv # else, should be of the format: traj_0.csv, traj_1.csv, traj_2.csv, ..., traj_n.csv

# Paths 
folder_path_main = "data_traj"

folders_compare = {
    "ood_gradual_edge_to_outside_set1": "rollout_traj/ood_gradual_edge_to_outside/set1",
    "ood_gradual_edge_to_outside_set2": "rollout_traj/ood_gradual_edge_to_outside/set2",
    "ood_between_four_circles": "rollout_traj/ood_between_four_circles",
    "In-distribution": "rollout_traj/in_dis",
    "distractor": "rollout_traj/distractor",
    "ood_square_middle_interpolate": "rollout_traj/ood_square_middle_interpolate",
}

plot_all_distance_chart = True
plot_similarity_closer_look_chart = True

# REMAPPING TRAJECTORIES, if needed, otherwise, make traj_remapping = None
traj_remapping = None
# do the mapping to bring the main trajectory to the order where we did rollout
# T12 should be T5,...., T5 should be T12
# T24 should be T19, .... T19 should be T24
traj_remapping = {
    12: 5,
    11: 6,
    10: 7,
    9: 8,
    8: 9,
    7: 10,
    6: 11,
    5: 12,
    24: 19,
    23: 20,
    22: 21,
    21: 22,
    20: 23,
    19: 24,
}

# folders_compare = {
#     "Random-place": "rollout_traj/random_place",
#     "Controlled-OoD": "rollout_traj/out_dis",
#     "In-distribution": "rollout_traj/in_dis"
# }




def compute_pointwise_distance(traj1, traj2):
    distances, aligned_points = [], []
    for point in traj1:
        dists = np.linalg.norm(traj2 - point, axis=1)
        closest_idx = np.argmin(dists)
        distances.append(dists[closest_idx])
        aligned_points.append(traj2[closest_idx])
    return np.array(distances), np.array(aligned_points)

def compute_average_distance(distances):
    return np.mean(distances)

# Loading function
def load_trajectories(folder, traj_prefix_is_timestamp=False):
    traj_files = []
    if traj_prefix_is_timestamp:
        traj_files = sorted([f for f in os.listdir(folder) if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_traj\.csv$", f)],
                            key=lambda x: re.findall(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', x)[0])
        print("Using timestamp-based naming convention for trajectory files.")
        print(f"Found {len(traj_files)} trajectory files after regex filtering.")
        print("Their names are:")
        for file in traj_files:
            print(file)
    else:
        traj_files = sorted([f for f in os.listdir(folder) if re.match(r"traj_\d+\.csv$", f)],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
        print("Using index-based naming convention for trajectory files.")
        print(f"Found {len(traj_files)} trajectory files after regex filtering.")
        print("Their names are:")
        for file in traj_files:
            print(file)
    # print the number of files found
    print(f"Found {len(traj_files)} trajectory files in {folder}.")
    trajectories = []
    for file in traj_files:
        data = pd.read_csv(os.path.join(folder, file), header=None)
        if data.shape[1] < 3:
            print(f"Skipping file {file} due to insufficient columns.")
            continue
        # load only the first three columns since they are x, y, z, the rest are orientation
        trajectories.append(data.iloc[:, :3].values)
    return trajectories

main_trajs = load_trajectories(folder_path_main, traj_prefix_is_timestamp=False)
if traj_remapping:
    # Step 1: Sort the files in the main folder
    remapped_files = []
    for i in range(len(main_trajs)):
        target_index = traj_remapping.get(i, i)  # Use mapped index if exists, else identity
        remapped_filename = main_trajs[target_index]
        remapped_files.append(remapped_filename)

    # Step 4: Use this list however you need
    print("\nFiles reordered according to mapping:")
    for i, fname in enumerate(remapped_files):
        print(f"T{i} → {fname}")

    main_trajs = remapped_files

if plot_similarity_closer_look_chart:
    for label, folder in folders_compare.items():
        print(f"Loading data from {folder}")
        target_trajs = load_trajectories(folder, traj_prefix_is_timestamp=rollout_traj_prefix_is_timestamp)
        num_targets = len(target_trajs)
        n_cols = 2
        n_rows = int(np.ceil(num_targets / n_cols))
        assert n_rows > 0, "Probably no data in the folder, please check the folder name and path"
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for idx, (target_traj, ax) in enumerate(zip(target_trajs, axes)):
            avg_distances, traj2_aligned_list = [], []

            for main_traj in main_trajs:
                distances, traj2_aligned = compute_pointwise_distance(target_traj, main_traj)
                avg_distances.append(compute_average_distance(distances))
                traj2_aligned_list.append(traj2_aligned)

            sorted_idx = np.argsort(avg_distances)
            best_idx, second_best_idx = sorted_idx[:2]

            best_dist = avg_distances[best_idx]
            second_best_dist = avg_distances[second_best_idx]

            distances_best_vs_second, _ = compute_pointwise_distance(
                traj2_aligned_list[best_idx], traj2_aligned_list[second_best_idx])
            avg_dist_best_vs_second = compute_average_distance(distances_best_vs_second)

            similarity_scores = [
                1 - best_dist / avg_dist_best_vs_second,
                1 - second_best_dist / avg_dist_best_vs_second,
                1 - avg_dist_best_vs_second / avg_dist_best_vs_second
            ]

            columns_labels = ['Target-Closest', 'Target-2nd Closest', 'Closest-2nd Closest']

            bars = ax.bar(columns_labels, similarity_scores, color='darkblue', alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Similarity Score (0-1)', color='darkblue')

            for bar in bars:
                height = bar.get_height()
                label_position = max(height, 0.0)
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, label_position),
                            xytext=(0, 1),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='darkblue')

            ax2 = ax.twinx()
            distances_values = [best_dist, second_best_dist, avg_dist_best_vs_second]
            ax2.plot(columns_labels, distances_values, 'o-', color='orange', markersize=10, linewidth=2.5)
            ax2.set_ylabel('Avg Dist. (m)', color='orange')

            for col, val in zip(columns_labels, distances_values):
                ax2.annotate(f'{val:.1f}', xy=(col, val), xytext=(0, 5), textcoords='offset points', fontsize=8, ha='center', color='orange')

            ax.text(0.02, 0.9, f'Traj #{idx}', transform=ax.transAxes,
                    ha='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            ax.grid(True)

        fig.suptitle(f"Similarity & Dist (Sim = 1 - Dist / Dist-2ndClosestToClosest) for {label}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save the figure
        fig_path = os.path.join('./similarity-plots', f"similarity_closest_and_2ndclosest_{label.replace(' ', '_')}.png")
        fig.savefig(fig_path, dpi=300)
        print(f"Saved plot to {fig_path}")


if plot_all_distance_chart:
    for label, folder in folders_compare.items():
        target_trajs = load_trajectories(folder, traj_prefix_is_timestamp=rollout_traj_prefix_is_timestamp)
        num_targets = len(target_trajs)
        n_cols = 2
        n_rows = int(np.ceil(num_targets / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for idx, (target_traj, ax) in enumerate(zip(target_trajs, axes)):
            avg_distances, traj2_aligned_list = [], []

            for main_traj in main_trajs:
                distances, traj2_aligned = compute_pointwise_distance(target_traj, main_traj)
                avg_distances.append(compute_average_distance(distances))
                traj2_aligned_list.append(traj2_aligned)

            sorted_idx = np.argsort(avg_distances)
            best_idx, second_best_idx = sorted_idx[:2]

            _, traj2_aligned_best = compute_pointwise_distance(target_traj, main_trajs[best_idx])
            _, traj2_aligned_second_best = compute_pointwise_distance(target_traj, main_trajs[second_best_idx])

            distances_best_vs_second, _ = compute_pointwise_distance(
                traj2_aligned_best, traj2_aligned_second_best)
            avg_dist_best_vs_second = compute_average_distance(distances_best_vs_second)

            similarity_scores = [1 - (dist / avg_dist_best_vs_second) for dist in avg_distances]
            similarity_scores = np.clip(similarity_scores, 0, 1)

            traj_labels = [f'T-{i}' for i in range(len(avg_distances))]

            bars = ax.bar(traj_labels, similarity_scores, color='darkblue', alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Similarity Score (0-1)', color='darkblue')
            ax.set_xticklabels(traj_labels, rotation=90)

            # Adding data labels for similarity scores
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 1),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='darkblue')

            # Plot distances on the right y-axis
            ax2 = ax.twinx()
            ax2.plot(traj_labels, avg_distances, 'o-', color='orange', markersize=8, linewidth=2)
            ax2.set_ylabel('Avg Dist. (m)', color='orange')

            # Adding data labels for distances
            for label_dist, val_dist in zip(traj_labels, avg_distances):
                ax2.annotate(f'{val_dist:.1f}', xy=(label_dist, val_dist), xytext=(0, 5),
                            textcoords='offset points', fontsize=8, ha='center', color='orange')

            ax.text(0.02, 0.9, f'Target Traj #{idx}', transform=ax.transAxes,
                    ha='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            ax.grid(True)

        fig.suptitle(f"Similarity & Distance (Sim = 1 - Dist / Dist-2ndClosestToClosest) for {label}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save closer look chart
        fig_path = os.path.join('./similarity-plots', f"dist_and_similarity_distribution_{label.replace(' ', '_')}.png")
        fig.savefig(fig_path, dpi=300)
        print(f"Saved closer look plot to {fig_path}")

plt.show()