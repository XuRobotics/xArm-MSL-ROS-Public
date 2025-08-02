#!/bin/bash

SESSION_NAME="realsense_cameras"

# Start new tmux session
tmux new-session -d -s $SESSION_NAME

# Launch first ROS file in first pane
tmux send-keys -t $SESSION_NAME "roslaunch ./launch_three_camera_utils/launch_three_realsense_cameras_first.launch" C-m

# Split the window horizontally and launch second ROS file
tmux split-window -h -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME:0.1 "sleep 1; roslaunch ./launch_three_camera_utils/launch_three_realsense_cameras_second.launch" C-m

# Split the second pane vertically and launch third ROS file
tmux split-window -v -t $SESSION_NAME:0.1
tmux send-keys -t $SESSION_NAME:0.2 "sleep 2; roslaunch ./launch_three_camera_utils/launch_three_realsense_cameras_third.launch" C-m

# # Create a fourth pane to issue a kill command
# tmux split-window -v -t $SESSION_NAME:0.0
# # give kill command but wait till the user presses enter
# tmux send-keys -t $SESSION_NAME:0.3 "echo 'Press Enter to kill all three cameras'" C-m
# tmux send-keys -t $SESSION_NAME:0.3 "read" C-m
# # tmux send-keys -t $SESSION_NAME:0.3 "tmux kill-session -t $SESSION_NAME" C-m
# # Attach to the tmux session
# tmux attach-session -t $SESSION_NAME:0.3

# Add window to easily kill all processes
tmux split-window -v -t $SESSION_NAME:0.0
tmux send-keys -t $SESSION_NAME "tmux kill-session -t ${SESSION_NAME}"

# Adjust layout to evenly distribute panes
tmux select-layout -t $SESSION_NAME tiled

tmux select-window -t $SESSION_NAME
tmux -2 attach-session -t $SESSION_NAME
