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

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# ---------------------------------------
# Overview of the code's functionality:
# - Adjustable parameters for similarity and distance visualization
# - Loading trajectory data from CSV files
# - Calculating minimum pointwise distances between trajectories
# - Computing similarity scores based on distances
# - Visualizing similarity scores as bar plots and distances as smoothed line plots
# - Enhancing plot readability and clarity with labels and annotations
# ---------------------------------------

# Adjustable parameters to define similarity thresholds and distance visualization limits
max_dist_for_similarity = {"In-distribution": 15, "Random-place": 15, "Controlled-OoD": 15}
max_dist_for_raw_plot = {"In-distribution": 100, "Random-place": 100, "Controlled-OoD": 100}

# Paths for main trajectory and trajectories used for comparison
folder_path_main = "data_traj"
folders_compare = {
    "In-distribution": "rollout_traj/in_dis",
    "Controlled-OoD": "rollout_traj/out_dis",
    "Random-place": "rollout_traj/random_place"
}

# Function to load trajectories from specified folder
def load_trajectories(folder):
    # List CSV files sorted numerically based on their naming pattern
    traj_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")],
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    trajectories = []
    # Iterate over files to load trajectory data
    # print loading data from
    print(f"Loading data from {folder}")
    for file in traj_files:
        data = pd.read_csv(os.path.join(folder, file), header=None)
        if data.shape[1] < 3:
            continue  # Skip files with insufficient columns
        trajectories.append(data.iloc[:, :3].values)
    return trajectories

# Compute the minimum distance from each point in traj1 to points in traj2
def compute_pointwise_distance(traj1, traj2):
    distances = []
    for point in traj1:
        closest_dist = np.min(np.linalg.norm(traj2 - point, axis=1)) # closest distance to any point in traj2 (to handle different lengths, sample rate, or non-synced trajectories)
        distances.append(closest_dist)
    return distances

# Load the main trajectories for comparison
main_trajs = load_trajectories(folder_path_main)

# Iterate through each comparison scenario (in-distribution, out-distribution, random-place)
for label, folder in folders_compare.items():
    
    # Load target trajectories to compare against main trajectories
    target_trajs = load_trajectories(folder)

    # Setup the plotting grid
    num_targets = len(target_trajs)
    n_cols = 2
    n_rows = int(np.ceil(num_targets / n_cols))

    # Create figure for similarity scores
    fig_sim, axes_sim = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
    axes_sim = axes_sim.flatten()

    # Loop over each target trajectory for comparison
    for idx, (target_traj, ax) in enumerate(zip(target_trajs, axes_sim)):
        similarity_scores = []
        distances_cropped = []

        # Compute similarity scores and cropped distances for visualization
        for main_traj in main_trajs:
            distances = compute_pointwise_distance(target_traj, main_traj)
            sorted_distances = sorted(distances)
            cutoff = int(len(sorted_distances) * 0.95) # use 95% of the points to calculate the mean distance, robust to outliers
            mean_distance = np.mean(sorted_distances[:cutoff])

            # Normalize similarity score based on max allowed distance
            max_dist_sim = max_dist_for_similarity[label]
            similarity_score = max(0, 1 - mean_distance / max_dist_sim)
            similarity_scores.append(similarity_score)

            # Crop distance values for clearer visualization in plots
            mean_distance_cropped = min(mean_distance, max_dist_for_raw_plot[label])
            distances_cropped.append(mean_distance_cropped)

        # Plot similarity scores as bars
        bars = ax.bar(range(len(main_trajs)), similarity_scores, color=f'C{idx}', alpha=0.9, bottom=-0.1)

        # Annotate the bar with the highest similarity score with the corresponding distance
        max_sim_idx = np.argmax(similarity_scores)
        max_bar = bars[max_sim_idx]
        max_dist = distances_cropped[max_sim_idx]
        ax.annotate(f'{max_dist:.1f} mm',
                    xy=(max_bar.get_x() + max_bar.get_width() / 2, max_bar.get_height() - 0.1),
                    xytext=(0, 1),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

        # Overlay smoothed line plot for distances
        x_original = np.arange(len(main_trajs))
        x_smooth = np.linspace(x_original.min(), x_original.max(), len(main_trajs) * 2)
        spline = make_interp_spline(x_original, distances_cropped, k=1)
        distances_smooth = spline(x_smooth)

        ax2 = ax.twinx()
        ax2.plot(x_smooth, distances_smooth, color='blue', alpha=0.4, linewidth=1.2)
        ax2.set_ylim(0, max_dist_for_raw_plot[label])
        ax2.set_ylabel("Distance (mm)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Setting axes limits and labels
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similarity Score")

        # Annotate subplot with trajectory number
        ax.text(0.05, 0.9, f"Traj #{idx}", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.grid(True)

    # Hide unused subplots to clean up visualization
    for ax in axes_sim[num_targets:]:
        ax.axis('off')

    # Finalize figure with titles and layout adjustments
    fig_sim.suptitle(f"{label} Trajectories", fontsize=16)
    fig_sim.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig_sim.text(0.5, 0.01, "Main Trajectory Index", ha='center')

# Display all plots
plt.show()