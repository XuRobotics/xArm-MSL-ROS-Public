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
from scipy.interpolate import make_interp_spline
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D
import re

rollout_traj_prefix_is_timestamp = True # If True, should be of the format e.g.: 2025-03-31-03-36-42_traj.csv # else, should be of the format: traj_0.csv, traj_1.csv, traj_2.csv, ..., traj_n.csv

# Paths for main trajectory and trajectories used for comparison
folder_path_main = "data_traj"

folders_compare = {
    "ood_gradual_edge_to_outside_set1": "rollout_traj/ood_gradual_edge_to_outside/set1",
    "ood_gradual_edge_to_outside_set2": "rollout_traj/ood_gradual_edge_to_outside/set2",
    "ood_between_four_circles": "rollout_traj/ood_between_four_circles",
    "In-distribution": "rollout_traj/in_dis",
    "distractor": "rollout_traj/distractor",
    "ood_square_middle_interpolate": "rollout_traj/ood_square_middle_interpolate",
}


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

main_trajs = load_trajectories(folder_path_main)
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
    
def compute_pointwise_distance(traj1, traj2):
    distances = []
    aligned_points = []
    for point in traj1:
        dists = np.linalg.norm(traj2 - point, axis=1)
        closest_idx = np.argmin(dists)
        distances.append(dists[closest_idx])
        aligned_points.append(traj2[closest_idx])
    return np.array(distances), np.array(aligned_points)

def compute_average_distance(distances):
    return np.mean(distances)


# Number of examples to plot
num_examples = 3  

for label, folder in folders_compare.items():
    target_trajs = load_trajectories(folder, rollout_traj_prefix_is_timestamp)
    num_targets = len(target_trajs)
    
    if num_targets < num_examples:
        num_examples = num_targets  # Adjust if there are fewer than 3 targets

    for i in range(num_examples):
        target_traj = target_trajs[i]

        avg_distances, traj2_aligned_list = [], []

        # Compute average distances and aligned trajectories
        for main_traj in main_trajs:
            distances, traj2_aligned = compute_pointwise_distance(target_traj, main_traj)
            avg_distances.append(compute_average_distance(distances))
            traj2_aligned_list.append(traj2_aligned)

        sorted_idx = np.argsort(avg_distances)
        best_idx, second_best_idx = sorted_idx[:2]

        closest_traj = main_trajs[best_idx]  # Full closest trajectory
        second_closest_traj = main_trajs[second_best_idx]  # Full second-closest trajectory

        aligned_closest = traj2_aligned_list[best_idx]  # Matched closest trajectory
        aligned_second_closest = traj2_aligned_list[second_best_idx]  # Matched second-closest trajectory

        fig_3d = plt.figure(figsize=(8, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Line plot for full target trajectory (green, **transparent**)
        ln_target, = ax_3d.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], 
                                'g-', alpha=0.6, linewidth=2)

        # Line plot for full closest trajectory (blue, transparent)
        ln_closest_full, = ax_3d.plot(closest_traj[:, 0], closest_traj[:, 1], closest_traj[:, 2], 
                                      'b-', alpha=0.3, linewidth=2)

        # Line plot for full second closest trajectory (red, transparent)
        ln_second_closest_full, = ax_3d.plot(second_closest_traj[:, 0], second_closest_traj[:, 1], second_closest_traj[:, 2], 
                                             'r-', alpha=0.3, linewidth=2)

        # Scatter plot for target trajectory (green spheres, **slightly transparent**)
        sc_target = ax_3d.scatter(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], 
                                  c='green', marker='o', alpha=0.5, s=40)

        # Scatter plot for matched closest trajectory (blue, **larger size**)
        sc_closest_matched = ax_3d.scatter(aligned_closest[:, 0], aligned_closest[:, 1], aligned_closest[:, 2], 
                                           c='blue', marker='o', alpha=1.0, s=50)

        # Scatter plot for matched second closest trajectory (red, **slightly smaller**)
        sc_second_closest_matched = ax_3d.scatter(aligned_second_closest[:, 0], aligned_second_closest[:, 1], aligned_second_closest[:, 2], 
                                                  c='red', marker='o', alpha=1.0, s=30)

        # Mark start (first point) and goal (last point) on the target trajectory, **3x larger**
        sc_start = ax_3d.scatter(target_traj[0, 0], target_traj[0, 1], target_traj[0, 2], 
                                 c='black', marker='o', s=300, edgecolors='white', label="Start")
        sc_goal = ax_3d.scatter(target_traj[-1, 0], target_traj[-1, 1], target_traj[-1, 2], 
                                c='black', marker='X', s=300, edgecolors='white', label="Rollout Stop")

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title(f"3D Trajectories for Example {i+1} - {label}")

        # Add legend inside each figure (upper right)
        ax_3d.legend([
            ln_target, sc_target, 
            ln_closest_full, ln_second_closest_full, 
            sc_closest_matched, sc_second_closest_matched, 
            sc_start, sc_goal
        ], [
            "Target Traj (Line)", "Target Traj (Scatter)", 
            "Closest Traj (Full)", "2nd Closest Traj (Full)", 
            "Closest Traj (Matched)", "2nd Closest Traj (Matched)", 
            "Start", "Goal"
        ], loc="upper right", fontsize=10)

plt.show()
