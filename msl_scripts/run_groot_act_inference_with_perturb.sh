#!/bin/bash

SESSION_NAME="groot_act_session"

# Kill existing session if it exists
tmux has-session -t $SESSION_NAME 2>/dev/null && tmux kill-session -t $SESSION_NAME

# === Window 1: Groot Eval ===
tmux new-session -d -s $SESSION_NAME -n "Groot Eval"
tmux send-keys -t $SESSION_NAME:"Groot Eval" 'condaactivategroot && cd /home/xarm/xArm-MSL-ROS/msl_scripts/model_inference' C-m
tmux send-keys -t $SESSION_NAME:"Groot Eval" 'python eval_groot_xarm_pick_place.py'

# === Window 2: ACT Training ===
tmux new-window -t $SESSION_NAME -n "ACT Training"
tmux send-keys -t $SESSION_NAME:"ACT Training" 'condaactivateact && cd /home/xarm/act' C-m
tmux send-keys -t $SESSION_NAME:"ACT Training" 'python imitate_episodes.py   --eval   --task_name xarm_pick_place   --ckpt_dir /home/xarm/bags/msl_bags/act_checkpoints/absolute_action_run4   --policy_class ACT   --chunk_size 100   --batch_size 16   --num_epochs 50000   --lr 1e-5   --dim_feedforward 3200   --hidden_dim 512   --seed 0   --kl_weight 0'

# === Window 3: Domain Perturbation ===
tmux new-window -t $SESSION_NAME -n "DomainPerturb"
tmux send-keys -t $SESSION_NAME:"DomainPerturb" 'condaactivategroot && cd /home/xarm/xArm-MSL-ROS/msl_scripts' C-m
tmux send-keys -t $SESSION_NAME:"DomainPerturb" 'python domain_generalization_create_perturbed_images.py'

# Create new window and get Pane 0's ID
PANE0=$(tmux new-window -P -F "#{pane_id}" -t $SESSION_NAME -n "Plotting")
# Split Pane 0 vertically, get new pane's ID (Pane 1)
PANE1=$(tmux split-window -v -P -F "#{pane_id}" -t $PANE0)
# Split Pane 0 vertically again, get new pane's ID (Pane 2)
PANE2=$(tmux split-window -v -P -F "#{pane_id}" -t $PANE0)

tmux send-keys -t $PANE0 'condaactivategroot && cd /home/xarm/xArm-MSL-ROS/msl_scripts/model_inference' C-m
# tmux send-keys -t $PANE0 C-l
tmux send-keys -t $PANE0 'python trajectory_plot_3D_rollout_act.py '

tmux send-keys -t $PANE1 'condaactivategroot && cd /home/xarm/xArm-MSL-ROS/msl_scripts/model_inference' C-m
# tmux send-keys -t $PANE1 C-l
tmux send-keys -t $PANE1 'python trajectory_similarity_plot_3D_rollout_groot_act.py --for_act True '

tmux send-keys -t $PANE2 'condaactivategroot && cd /home/xarm/xArm-MSL-ROS/msl_scripts/model_inference' C-m
# tmux send-keys -t $PANE2 C-l
tmux send-keys -t $PANE2 'python trajectory_similarity_plot_3D_rollout_groot_act.py --for_act False '

# Arrange panes in vertical layout (even-vertical)
tmux select-layout -t $SESSION_NAME:"Plotting" even-vertical

# === Window 5: Kill All Sessions ===
tmux new-window -t $SESSION_NAME -n "Kill"
kill_cmd="echo 'Press Enter to kill all tmux sessions'; read; tmux kill-server"
tmux send-keys -t $SESSION_NAME:"Kill" "$kill_cmd" C-m

# Attach to session
tmux select-window -t $SESSION_NAME:0
tmux attach-session -t $SESSION_NAME