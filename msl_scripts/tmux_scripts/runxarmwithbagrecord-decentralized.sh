#!/bin/bash

SESSION_NAME="runxarmwithbagrecord"

# Start new tmux session
tmux new-session -d -s $SESSION_NAME

# Launch first ROS file in first pane
tmux send-keys -t $SESSION_NAME "recordbagxarmdemo" C-m

# Split the window horizontally and launch second ROS file
tmux split-window -h -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME:0.1 "runxarmscript" C-m

# Split the second pane vertically and launch third ROS file
tmux split-window -v -t $SESSION_NAME:0.1
tmux send-keys -t $SESSION_NAME:0.2 "" C-m

# # Create a fourth pane to issue a kill command
# tmux split-window -v -t $SESSION_NAME:0.0
# # give kill command but wait till the user presses enter
# tmux send-keys -t $SESSION_NAME:0.3 "echo 'Press Enter to kill all three cameras'" C-m
# tmux send-keys -t $SESSION_NAME:0.3 "read" C-m
# # tmux send-keys -t $SESSION_NAME:0.3 "tmux kill-session -t $SESSION_NAME" C-m
# # Attach to the tmux session
# tmux attach-session -t $SESSION_NAME:0.3


# Create a fourth pane to issue a kill command
tmux split-window -v -t $SESSION_NAME:0.0

# Provide instruction and wait for user confirmation before killing
kill_cmd="echo 'Press Enter to kill all processes'; read; \
          tmux send-keys -t $SESSION_NAME:0.0 C-c; \
          tmux send-keys -t $SESSION_NAME:0.1 C-c; \
          sleep 1; \
          tmux kill-session -t $SESSION_NAME"

tmux send-keys -t $SESSION_NAME:0.3 "$kill_cmd" C-m


# Adjust layout to evenly distribute panes
tmux select-layout -t $SESSION_NAME tiled

tmux select-window -t $SESSION_NAME
tmux -2 attach-session -t $SESSION_NAME:0.3
