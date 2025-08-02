# xArm-MSL-ROS

## 🟢Overview

This repository provides code related to working on the Ufactory xArm manipulators in Stanford MSL. It includes:

- Model training and find-tuning suppport (data preparation, training scripts, etc):
  - GROOT N1.5
  - ACT (Action Chunking with Transformers)
  - Diffusion Policy 

- Data collection and visualization utilities using ROS, RealSense, and VRPN for demonstration recording.

- A modified version of the official xArm Python SDK adapted for integration with MSL workflows.

## 🟢Table of Contents

- [xArm-MSL-ROS](#xarm-msl-ros)
  - [🟢Overview](#overview)
  - [🟢Table of Contents](#table-of-contents)
  - [🟢Groot N1.5: Data Preparation \& Fine-Tuning](#groot-n15-data-preparation--fine-tuning)
    - [Install the Right Version of GR00T](#install-the-right-version-of-gr00t)
    - [Data Preparation](#data-preparation)
    - [Fine-Tuning the Model](#fine-tuning-the-model)
    - [Resuming Training](#resuming-training)
    - [Running Inference](#running-inference)
      - [Visualization](#visualization)
      - [Inference Without Robot Data](#inference-without-robot-data)
  - [🟢ACT: Data Preparation \& Training](#act-data-preparation--training)
    - [Install the Right Version of ACT](#install-the-right-version-of-act)
    - [Data Preparation](#data-preparation-1)
    - [Training the Model](#training-the-model)
      - [Resume Training](#resume-training)
      - [Train with Relative Actions](#train-with-relative-actions)
    - [Running Inference](#running-inference-1)
      - [Visualization](#visualization-1)
  - [🟢Diffusion Policy: xArm Experiments](#diffusion-policy-xarm-experiments)
    - [Run Diffusion Policy](#run-diffusion-policy)
    - [After Experiments](#after-experiments)
  - [🟢MSL Data Collection with OptiTrack](#msl-data-collection-with-optitrack)
  - [🟢xArm-Python-SDK](#xarm-python-sdk)

---

## 🟢Groot N1.5: Data Preparation & Fine-Tuning

> PS: Trained model's **checkpoints** for **MSL xArm** can be found at:
> [Groot Checkpoints](https://drive.google.com/drive/folders/1Pqw7ueTiyrOWk6qdn63Rsd0vm-RZXs39?usp=sharing)

### Install the Right Version of GR00T

Use [my GR00T repo](https://github.com/XuRobotics/Isaac-GR00T), which differs from the [original GR00T repo](https://github.com/NVIDIA/Isaac-GR00T):

* Custom config specifying action representation and action horizon

### Data Preparation

Inside Docker:

```bash
dockerrunxarm
cd /xArm-MSL-ROS/msl_scripts/groot_data_prepare
python convert_data_for_groot.py
python generate_info_json.py
python generate_episode_task_json.py
python generate_modality_json.py
```

**IMPORTANT NOTES:**

* Gripper value **is scaled to 0–1** here by `convert_data_for_groot.py`. It is **no longer** 0–850 (raw data from xArm API).
* The state/action inputs format: `[orientation (6D), position, gripper (0–1)]`

### Fine-Tuning the Model

Outside Docker:

```bash
conda activate gr00t
```

Start fine-tuning:

```bash
python scripts/gr00t_finetune.py \
--dataset-path ~/bags/msl_bags/converted_groot_data_absolute  \
--output-dir ~/bags/msl_bags/groot_checkpoints/xarm_pick_place_absolute_pose_run6_batch_16_horizon_100  \
--data-config xarm_dualcam_h100  \
--embodiment-tag oxe_droid  \
--num-gpus 1  \
--no-tune_diffusion_model  \
--max-steps 200000 \
--batch-size 16
```

### Resuming Training

1. Edit the file:

   ```
   /home/xarm/anaconda3_outside_docker/envs/gr00t/lib/python3.10/site-packages/transformers/trainer.py
   ```

   Change this line:

   ```python
   checkpoint_rng_state = torch.load(rng_file, weights_only=True)
   ```

   To:

   ```python
   checkpoint_rng_state = torch.load(rng_file, weights_only=False)
   ```

2. Add the `--resume` flag:

```bash
python scripts/gr00t_finetune.py \
--dataset-path ~/bags/msl_bags/converted_groot_data_absolute  \
--output-dir ~/bags/msl_bags/groot_checkpoints/xarm_pick_place_absolute_pose_run6_batch_16_horizon_100  \
--data-config xarm_dualcam_h100  \
--embodiment-tag oxe_droid  \
--num-gpus 1  \
--no-tune_diffusion_model  \
--max-steps 200000 \
--batch-size 16 \
--resume
```

### Running Inference

Inside Docker:

```bash
python extract_images_poses_from_bags.py
```

> **NOTE**: State/action inputs are in the format: `[position, orientation (6D), gripper (0–850)]`

Outside Docker:

```bash
python eval_groot_xarm_pick_place.py
```

#### Visualization

3D plot:

```bash
python trajectory_plot_3D_rollout_groot.py
```

Similarity plot:

Edit `trajectory_similarity_plot_3D_rollout_groot_act.py` and set:

```python
for_act = False
```

Then:

```bash
python trajectory_similarity_plot_3D_rollout_groot_act.py
```

#### Inference Without Robot Data

```bash
python scripts/eval_policy.py \
  --model_path ~/bags/msl_bags/groot_checkpoints/xarm_pick_place_absolute_pose/checkpoint-25000 \
  --data_config xarm_dualcam_h100 \
  --dataset_path ~/bags/msl_bags/converted_groot_data \
  --embodiment_tag oxe_droid \
  --video_backend torchvision_av \
  --modality_keys single_arm gripper \
  --plot
```

---

## 🟢ACT: Data Preparation & Training

> PS: Trained model's **checkpoints** for **MSL xArm** can be found at:
> [ACT Checkpoints](https://drive.google.com/drive/folders/1p3TibY0OldiD5uDp252AJTS7c2pgV0n6?usp=sharing)

### Install the Right Version of ACT

Use [my ACT repo](https://github.com/XuRobotics/act), which differs from the [original ACT repo](https://github.com/tonyzhaozh/act):

* Action representation = end-effector pose + gripper (10-dim) instead of joint-space (14-dim)
* Updated data loader for custom observations

### Data Preparation

Inside Docker:

```bash
dockerrunxarm
cd /xArm-MSL-ROS/msl_scripts/act_data_prepare
python convert_data_for_act.py
python generate_act_config.py
```

**IMPORTANT NOTES:**

* Gripper values **are NOT scaled**; they remain **0–850**
* Input format: `[position, orientation (6D), gripper (0–850)]`

### Training the Model

Outside Docker:

```bash
conda activate aloha
cd act
```

Train:

```bash
python imitate_episodes.py \
  --task_name xarm_pick_place \
  --ckpt_dir /home/xarm/bags/msl_bags/act_checkpoints/absolute_action_run3 \
  --policy_class ACT \
  --chunk_size 100 \
  --batch_size 16 \
  --num_epochs 50000 \
  --lr 1e-5 \
  --dim_feedforward 3200 \
  --hidden_dim 512 \
  --seed 0 \
  --kl_weight 0
```

#### Resume Training

```bash
--resume_ckpt_path /home/xarm/bags/msl_bags/act_checkpoints/absolute_action/policy_epoch_30600_seed_0.ckpt
```

#### Train with Relative Actions

1. Set relative flag in `convert_data_for_act.py`
2. Update `--ckpt_dir`
3. Edit `constants.py` → `REAL_DATASET_DIR` to point to relative dataset

### Running Inference

Inside Docker:

```bash
python extract_images_poses_from_bags.py
```

Outside Docker:

```bash
python imitate_episodes.py \
  --eval \
  --task_name xarm_pick_place \
  --ckpt_dir /home/xarm/bags/msl_bags/act_checkpoints/absolute_action_run4 \
  --policy_class ACT \
  --chunk_size 100 \
  --batch_size 16 \
  --num_epochs 50000 \
  --lr 1e-5 \
  --dim_feedforward 3200 \
  --hidden_dim 512 \
  --seed 0 \
  --kl_weight 0 \
  --inference_dataset_dir /home/xarm/bags/msl_bags/IMPORTANT-distribution-pick-and-place-raw-bags-30/extracted_images_and_pose
```

#### Visualization

3D plot:

```bash
python trajectory_plot_3D_rollout_act.py
```

Similarity plot:

Edit `trajectory_similarity_plot_3D_rollout_groot_act.py` and set:

```python
for_act = True
```

Then:

```bash
python trajectory_similarity_plot_3D_rollout_groot_act.py
```

---

## 🟢Diffusion Policy: xArm Experiments

Start Docker:

```bash
dockerrunxarm
```

> If container already exists and fails to start:

```bash
dockercommitxarm && dockerpushxarm && dockerrunxarmwithremove
```

Start `tmux` and open multiple windows:

```bash
tmux
Ctrl+b c   # to create new window
Ctrl+b n   # to switch to next window
```

**In separate windows:**

1. Start ROS:

```bash
roscore
```

2. Launch cameras:

```bash
cd xArm-MSL-ROS/
roslaunch msl_scripts/launch_two_realsense_cameras.launch
```

(Optional: use `rqt` to check camera feed, then close viewer.)

**Important: one person should remain at the robot's safety switch.**

Enable robot at:

```
http://192.168.1.219:18333
```

### Run Diffusion Policy

```bash
cd diffusion_policy/
conda-init
conda activate robodiff
python eval_xarm_msl_ros.py
```

### After Experiments

Outside Docker:

```bash
dockercommitxarm
dockerpushxarm
docker rm xarm
```

---

## 🟢MSL Data Collection with OptiTrack

```bash
cd ~/xArm-MSL-ROS
roslaunch msl_scripts/launch_two_realsense_cameras.launch
roslaunch vrpn_client_ros sample.launch
python ./xArm-Python-SDK/example/wrapper/xarm6/xarm_msl_ros.py
```

Check pose estimates:

```bash
rostopic echo /vrpn_client_node/drone1/pose
```

Visualize data:

```bash
rqt_image_view
```

Record demo data:

```bash
rosbag record -o xarm_demo \
  /robot_end_effector_pose \
  /wrist_camera/color/image_raw \
  /wrist_camera/color/camera_info \
  /fixed_camera/color/camera_info \
  /fixed_camera/color/image_raw \
  /tf /tf_static
```

---

## 🟢xArm-Python-SDK

See `README.md` in the `xArm-Python-SDK` subfolder for more details.
