"""
MIT License

Copyright (c) 2025 Xu Liu, Chengyang He, Gadiel Sznaier Camps at 
Multi-Robot Systems Lab, Stanford University (with Github Copilot)

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


import rosbag 
import os
import numpy as np
import zarr
import cv2
from cv_bridge import CvBridge
import imageio
import argparse
import time
from scipy.spatial.transform import Rotation as R

# Install necessary packages
try:
    import zarr
    import imageio
except ImportError:
    print("Installing required packages...")
    os.system("pip install zarr imageio[ffmpeg]")

def ensure_dir(directory, check=False):
    if not check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created for data store.")
    # if already exists, delete it with user's key in confirmation (double confirm)
    elif os.path.exists(directory):
        response = input(f"Directory '{directory}' already exists. Delete it? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -r {directory}")
        else:
            print("Exiting...")
            exit()
        os.makedirs(directory)
    else:
        # confirm to create the directory
        response = input(f"Directory '{directory}' does not exist. Create it? (y/n): ")
        if response.lower() == 'y':
            os.makedirs(directory)
            print(f"Directory '{directory}' created for data store.")

def process_rosbags(bags_folder, output_dir, consider_orientation):
    ensure_dir(output_dir, check=True)
    # sort bag files by their names    
    bag_files = [os.path.join(bags_folder, f) for f in os.listdir(bags_folder) if f.endswith(".bag")]
    # sort bag by stamp: xarm_demo_2025-03-19-10-34-00.bag, the stamp is 2025-03-19-10-34-00
    # print sorting bags according to the last part of the filename
    bag_files.sort(key=lambda x: x.split("_")[-1].split(".")[0])
    print(f"Found {len(bag_files)} bag files in {bags_folder}.")
    print("Bag files sorted by timestamp:")
    for bag_file in bag_files:
        print(bag_file)
    
    for bag_index, bag_file in enumerate(bag_files):
        print(f"Processing bag {bag_index}: {bag_file}")
        bag = rosbag.Bag(bag_file, "r")
        
        pose_msgs = []
        
        for topic, msg, t in bag.read_messages():
            if topic == '/robot_end_effector_pose':
                # record 3d position and orientation
                pose_msgs.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        
                
        # save the position in the pose message as a .csv file, add bag_index in the filename
        with open(os.path.join(output_dir, f"traj_{bag_index}.csv"), "w") as f:
            for pose in pose_msgs:
                if not consider_orientation:
                    # only save position
                    f.write(f"{pose[0]}, {pose[1]}, {pose[2]}\n")
                else:
                    # convert quaternion to euler angles
                    # from quat takes in [quat_x, quat_y, quat_z, quat_w] by default
                    r = R.from_quat([pose[3], pose[4], pose[5], pose[6]]) 
                    euler_angles = r.as_euler('xyz', degrees=False)
                    f.write(f"{pose[0]}, {pose[1]}, {pose[2]}, {euler_angles[0]}, {euler_angles[1]}, {euler_angles[2]}\n")
        bag.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_folder', type=str, default='/home/sam/bags/msl_bags/distribution_demo_pick_and_place',
                        help='Folder containing ROS bag files.')
    parser.add_argument('--output_dir', type=str, default='/home/sam/bags/msl_bags/traj_data_4_1',
                        help='Output directory to save synchronized data.')
    args = parser.parse_args()
    consider_orientation = True
    process_rosbags(args.bags_folder, args.output_dir, consider_orientation)