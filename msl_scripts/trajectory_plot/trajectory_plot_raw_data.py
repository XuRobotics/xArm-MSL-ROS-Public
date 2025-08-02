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

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


rollout_traj_prefix_is_timestamp = True # If True, should be of the format e.g.: 2025-03-31-03-36-42_traj.csv # else, should be of the format: traj_0.csv, traj_1.csv, traj_2.csv, ..., traj_n.csv

folder_path_main = "data_traj"

output_dir = "3D_raw_data"
os.makedirs(output_dir, exist_ok=True)


folders_compare = [
    # Previously light green->blue, now changed clearly to light blue -> blue
    ("rollout_traj/in_dis", "In-dis Trajectories", (0.7, 0.9, 1.0), (0, 0, 1.0)),

    # Previously purple, now clearly changed to shades of red
    ("rollout_traj/ood_gradual_edge_to_outside/set1", "ood_gradual_edge_to_outside-set1 Trajectories", (1.0, 0.7, 0.7), (0.8, 0, 0)),

    # ("rollout_traj/ood_gradual_edge_to_outside/set2", "ood_gradual_edge_to_outside-set2 Trajectories", (1.0, 0.7, 0.7), (0.8, 0, 0)),
    # make color slightly different from set1, make it magenta
    ("rollout_traj/ood_gradual_edge_to_outside/set2", "ood_gradual_edge_to_outside-set2 Trajectories", (1.0, 0.7, 0.7), (1.0, 0, 1.0)),
    
    # Previously light orange->red, now clearly changed to light orange -> orange
    ("rollout_traj/ood_between_four_circles", "ood_between_four_circles Trajectories", (1.0, 0.85, 0.6), (1.0, 0.5, 0)),
    
    # just purple -> purple, no change
    ("rollout_traj/distractor", "distractor Trajectories", (0.8, 0.6, 1.0), (0.8, 0.6, 1.0)),
    
    # cyan -> light cyan
    ("rollout_traj/ood_square_middle_interpolate", "ood_square_middle_interpolate Trajectories", (0.6, 1.0, 1.0), (0.6, 1.0, 1.0)),

]

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


# Helper functions
def interpolate_color(start_color, end_color, fraction):
    return tuple(s + fraction * (e - s) for s, e in zip(start_color, end_color))

def smooth_xyz(x, y, z, factor=2):
    idx = np.linspace(0, len(x) - 1, max(len(x) // factor, 4)).astype(int)
    x_sub, y_sub, z_sub = x.iloc[idx], y.iloc[idx], z.iloc[idx]

    points_new = np.linspace(0, len(idx) - 1, 200)
    x_smooth = make_interp_spline(range(len(idx)), x_sub)(points_new)
    y_smooth = make_interp_spline(range(len(idx)), y_sub)(points_new)
    z_smooth = make_interp_spline(range(len(idx)), z_sub)(points_new)

    
    return x_smooth, y_smooth, z_smooth

# Main trajectories
files_main = sorted(
    [f for f in os.listdir(folder_path_main) if re.match(r"traj_\d+\.csv$", f)],
    key=lambda x: int(re.findall(r'\d+', x)[0])
)
print(f"Found {len(files_main)} main trajectory files after regex filtering.")
print("Their names are:")
for file in files_main:
    print(file)
gray_color = (0.6, 0.6, 0.6)

if traj_remapping:
    # Step 1: Sort the files in the main folder
    remapped_files = []
    for i in range(len(files_main)):
        target_index = traj_remapping.get(i, i)  # Use mapped index if exists, else identity
        remapped_filename = files_main[target_index]
        remapped_files.append(remapped_filename)

    # Step 4: Use this list however you need
    print("\nFiles reordered according to mapping:")
    for i, fname in enumerate(remapped_files):
        print(f"T{i} → {fname}")

    files_main = remapped_files

# Plot comparisons separately
for folder_path, title, color_start, color_end in folders_compare:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)

    # Updated plot for main trajectories: green, dashed, thick, transparent
    green_color = (0.5, 0.5, 0.5)

    # for file in files_main:
    #     file_path = os.path.join(folder_path_main, file)
    #     data = pd.read_csv(file_path, header=None)
    #     if data.shape[1] < 3:
    #         continue

    #     x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
    #     spl_x, spl_y, spl_z = smooth_xyz(x, y, z)
    #     ax.plot(spl_x, spl_y, spl_z, color=green_color, alpha=0.5, linewidth=1)

    for i, file in enumerate(files_main):
        file_path = os.path.join(folder_path_main, file)
        data = pd.read_csv(file_path, header=None)
        if data.shape[1] < 3:
            continue

        x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
        spl_x, spl_y, spl_z = smooth_xyz(x, y, z)

        # Always draw full trajectory in base gray/green color
        ax.plot(spl_x, spl_y, spl_z, color=green_color, alpha=0.5, linewidth=1)

        # Highlighting logic
        highlight_set1 = "ood_gradual_edge_to_outside/set1" in folder_path and i == 17
        highlight_set2 = "ood_gradual_edge_to_outside/set2" in folder_path and i == 28
        highlight_distractor = "distractor" in folder_path and i == 15

        if highlight_set1 or highlight_set2 or highlight_distractor:
            n = len(spl_x)
            one_third = n // 2
            ax.plot(spl_x[:one_third], spl_y[:one_third], spl_z[:one_third],
                    color='green', linewidth=3, alpha=1.0)
            ax.scatter(spl_x[:one_third:20], spl_y[:one_third:20], spl_z[:one_third:20],
                       color='green', s=30, marker='o')



    # Plot comparison trajectories
    # ORIGINAL VERSION: traj_0.csv, traj_1.csv, traj_2.csv, ..., traj_n.csv
    print(f"Getting comparison trajectories from {folder_path} folder.")
    if not rollout_traj_prefix_is_timestamp:
        files_compare = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
    else:
        # NEW FORMAT: 2025-03-31-03-36-42_traj.csv, sort by the timestamp (e.g. 2025-03-31-03-36-42 is earlier than 2025-03-31-03-36-43)
        files_compare = sorted([f for f in os.listdir(folder_path) if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_traj\.csv$", f)],
                            key=lambda x: re.findall(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', x)[0])
    print (f"Found {len(files_compare)} comparison trajectory files after regex filtering.")
    print("Their names are:")
    for file in files_compare:
        print(file)
    n_compare = len(files_compare)

    for idx, file in enumerate(files_compare):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None)
        if data.shape[1] < 3:
            continue

        color = interpolate_color(color_start, color_end, idx / max(n_compare - 1, 1))
        x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
        spl_x, spl_y, spl_z = smooth_xyz(x, y, z)
        ax.plot(spl_x, spl_y, spl_z, color=color, alpha=0.7, linewidth=2)

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)
    ax.set_title(f"3D Trajectories Comparison\nGray: traj_x vs. Colored: {title}", fontsize=14)
    ax.set_facecolor('whitesmoke')


# Separate plot for main trajectories
fig_main = plt.figure(figsize=(10, 8))
ax_main = fig_main.add_subplot(111, projection='3d')
ax_main.view_init(elev=30, azim=45)

for file in files_main:
    file_path = os.path.join(folder_path_main, file)
    data = pd.read_csv(file_path, header=None)
    if data.shape[1] < 3:
        continue

    x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
    spl_x, spl_y, spl_z = smooth_xyz(x, y, z)
    ax_main.plot(spl_x, spl_y, spl_z, color=gray_color, alpha=0.8, linewidth=1)

ax_main.set_xlabel("X (mm)", fontsize=12)
ax_main.set_ylabel("Y (mm)", fontsize=12)
ax_main.set_zlabel("Z (mm)", fontsize=12)
ax_main.set_title("Main Trajectories (Gray)", fontsize=14)
ax_main.set_facecolor('whitesmoke')


# Plot all trajectories together
fig_all = plt.figure(figsize=(10, 8))
ax_all = fig_all.add_subplot(111, projection='3d')
ax_all.view_init(elev=30, azim=45)

# Plot main trajectories
for file in files_main:
    file_path = os.path.join(folder_path_main, file)
    data = pd.read_csv(file_path, header=None)
    if data.shape[1] < 3:
        continue

    x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
    spl_x, spl_y, spl_z = smooth_xyz(x, y, z)
    ax_all.plot(spl_x, spl_y, spl_z, color=gray_color, alpha=0.4, linewidth=1.5)

# Plot all comparison trajectories
for folder_path, _, color_start, color_end in folders_compare:
    # ORIGINAL VERSION: traj_0.csv, traj_1.csv, traj_2.csv, ..., traj_n.csv
    if not rollout_traj_prefix_is_timestamp:
        files_compare = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
    else:
        # NEW FORMAT: 2025-03-31-03-36-42_traj.csv, sort by the timestamp (e.g. 2025-03-31-03-36-42 is earlier than 2025-03-31-03-36-43)
        files_compare = sorted([f for f in os.listdir(folder_path) if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_traj\.csv$", f)],
                            key=lambda x: re.findall(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', x)[0])
    n_compare = len(files_compare)

    for idx, file in enumerate(files_compare):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None)
        if data.shape[1] < 3:
            continue

        color = interpolate_color(color_start, color_end, idx / max(n_compare - 1, 1))
        x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
        spl_x, spl_y, spl_z = smooth_xyz(x, y, z)
        ax_all.plot(spl_x, spl_y, spl_z, color=color, alpha=0.7, linewidth=2)

ax_all.set_xlabel("X (mm)", fontsize=12)
ax_all.set_ylabel("Y (mm)", fontsize=12)
ax_all.set_zlabel("Z", fontsize=12)
ax_all.set_title("All Trajectories Combined", fontsize=14)
ax_all.set_facecolor('whitesmoke')

plt.show()
