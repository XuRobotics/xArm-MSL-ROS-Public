"""
MIT License

Copyright (c) 2025 Xu Liu Multi-Robot Systems Lab, Stanford University (with Github Copilot)

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
from datetime import datetime
torch.manual_seed(0)
np.random.seed(0)
from torchvision import transforms
from PIL import Image
from matplotlib.lines import Line2D 
import glob
from scipy.spatial.transform import Rotation as R
import gc

# NOTE: See NETWORK ARCHITECTURE DIAGRAM in the end of this file for a visual representation of the model architecture.

# === Hyperparameters and Settings ===
use_data_augmentation = False # whether to apply data augmentation (e.g., random flips, rotations, translations)
flipping_actions = True  # whether to flip the action for the second start-goal pair
same_goal_positions = False # whether both start-goal pairs share the same goal location
num_inference_samples = 10
gif_skip_interval = 2
image_size = 64  # input image resolution
batch_size = 4096  # training batch size
learning_rate = 1e-3  # optimizer learning rate
T = 50  # number of diffusion steps
num_epochs = 2000  # number of training epochs
N = 100000  # number of training samples
embed_dimension_feature_vector = 64 # embedding dimension of feature vector, 64 or 256 or 1024
run_inference_only = False # If true, specify inference_model_path, where best_model.pt is saved
if run_inference_only:
    run_test_set_only = True # Skip training set inference and visualization, only run on test set images
else:
    run_test_set_only = False    
# inference_model_path = "/home/xarm/xArm-MSL-ROS/basic_diffusion_model/viz_outputs/2025-7-23/domain_generalization_diffusion_N_100000_epoch_2000_without_viewpoint_condition"
inference_model_path = "/home/xarm/xArm-MSL-ROS/basic_diffusion_model/viz_outputs/domain_generalization_diffusion_N_100000_epoch_5000_without_viewpoint_condition_2025-07-24_18-05-14"
custom_image_folder = "./custom_images/full_data"  
test_folder = "./custom_images/test_set_with_viewpoint_labels"
test_image_paths = []  # to hold paths of test images

include_pose_in_input = True  # Set to False if you do NOT want to condition on 3rd view camera pose


# util function to convert quaternion to 6D rotation representation
def quat_to_six_d(quat):
    """
    Converts a quaternion [x, y, z, w] to a 6D rotation representation (first 2 columns of rotation matrix).
    """
    rot_matrix = R.from_quat(quat).as_matrix()
    rot_6d = rot_matrix[:, :2].flatten()
    return rot_6d


# util function to automatically handle all parsed custom image inputs
def prepare_dataset(inputs, total_count):
    image_tensors, action_tensors, image_pose_tensors = [], [], []

    original_len = len(inputs)
    needed_repeats = (total_count + original_len - 1) // original_len

    if not use_data_augmentation:
        loaded_imgs = []
        loaded_actions = []
        loaded_poses = []
        for img_path, action, cur_view_deg in inputs:
            img_tensor = load_and_prepare_image(img_path, size=(W, H), augment=False)
            loaded_imgs.append(img_tensor)
            loaded_actions.append(torch.tensor(action, dtype=torch.float32))
            # loaded_poses.append(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, cur_view_deg * np.pi / 180.0], dtype=torch.float32))
            third_view_cam_pose = torch.tensor([np.sin(cur_view_deg * np.pi / 180.0), np.cos(cur_view_deg * np.pi / 180.0)], dtype=torch.float32)
            loaded_poses.append(third_view_cam_pose)
        # Repeat as needed
        images = torch.stack(loaded_imgs)
        actions = torch.stack(loaded_actions)
        poses = torch.stack(loaded_poses)
        images = images.repeat((needed_repeats, 1, 1, 1))[:total_count]
        actions = actions.repeat((needed_repeats, 1))[:total_count]
        poses = poses.repeat((needed_repeats, 1))[:total_count]
    else:
        cur_count = 0
        for _ in range(needed_repeats):
            cur_count += 1
            if cur_count % 100 == 1:
                print(f"doing augmentation, progress: {cur_count}/{needed_repeats} ({cur_count / needed_repeats * 100:.2f}%)")
            for img_path, action, cur_view_deg in inputs:
                img_tensor = load_and_prepare_image(img_path, size=(W, H), augment=True)
                image_tensors.append(img_tensor)
                action_tensors.append(torch.tensor(action, dtype=torch.float32))
                # image_pose_tensors.append(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, cur_view_deg * np.pi / 180.0], dtype=torch.float32))
                third_view_cam_pose = torch.tensor([np.sin(cur_view_deg * np.pi / 180.0), np.cos(cur_view_deg * np.pi / 180.0)], dtype=torch.float32)
                image_pose_tensors.append(third_view_cam_pose)

        images = torch.stack(image_tensors)[:total_count]
        actions = torch.stack(action_tensors)[:total_count]
        poses = torch.stack(image_pose_tensors)[:total_count]

    print(f"Prepared dataset with {len(images)} samples, {len(actions)} sample actions, and {len(poses)} sample poses.")

    # Shuffle
    perm = torch.randperm(total_count)
    images = images[perm]
    actions = actions[perm]
    poses = poses[perm]

    if include_pose_in_input:
        return images, actions, poses
    else:
        return images, actions

def view_num_to_deg(view_num, training_view_nums):
    """
    Converts a view number to its corresponding degree based on the training view numbers.
    """
    view_interval_deg = 360 / max(1, (max(training_view_nums) - min(training_view_nums)))  # handle case where all views are the same
    return (view_num - min(training_view_nums)) * view_interval_deg  # calculate the view degree based on the view number


# === Auto-load and parse image-action pairs from folder ===
def load_custom_images_from_folder(folder_path):
    supported_exts = ['*.png', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    image_paths = []
    for ext in supported_exts:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # sort images
    try:
        # if view1_left.png, view1_right.png, view1_forward.png, view2_left.png, etc.:
        image_paths.sort(key=lambda x: int(os.path.basename(x).split('view')[1].split('_')[0]))
        # further sort such that left, right, forward are in order
        image_paths.sort(key=lambda x: os.path.basename(x).split('_')[1])
        
        
    # if not, just sort by name
    except:
        image_paths.sort()
    
    parsed_inputs = []

    training_view_nums = []

    for path in image_paths:
        fname = os.path.basename(path).lower()

        # according to the file name, it has view1, view 2, ..., view 19, ..., view XX, etc, increase the camera yaw angle based on this
        if 'view' in fname:
            view_num = int(fname.split('view')[1].split('_')[0])
            training_view_nums.append(view_num)
        else:
            raise ValueError(f"Cannot determine view number from filename: {fname}")
    

    for path in image_paths:
        fname = os.path.basename(path).lower()

        if 'left' in fname:
            action = [-1.0, 0.0]
        elif 'right' in fname:
            action = [1.0, 0.0]
        elif 'up' in fname or 'forward' in fname:
            action = [0.0, -1.0]
        elif 'down' in fname or 'backward' in fname:
            action = [0.0, 1.0]
        else:
            raise ValueError(f"Cannot determine action from filename: {fname}")

        # according to the file name, it has view1, view 2, ..., view 19, ..., view XX, etc, increase the camera yaw angle based on this
        if 'view' in fname:
            view_num = int(fname.split('view')[1].split('_')[0])
            view_deg = view_num_to_deg(view_num, training_view_nums)
            print(f"View number {view_num} corresponds to {view_deg} degrees")
        else:
            raise ValueError(f"Cannot determine view number from filename: {fname}")

        parsed_inputs.append((path, action, view_deg))
    return parsed_inputs, training_view_nums


print("\n=== Loading training images ===")
print("Running inference on training set images is set as: ", not run_test_set_only)
# Run inference on training images
custom_image_inputs, training_view_nums = load_custom_images_from_folder(custom_image_folder)
print("Parsed inputs:")
for img_path, action, view_deg in custom_image_inputs:
    print(f"Image: {img_path}, Action: {action}, View: {view_deg} degrees")
# check if empty
if not custom_image_inputs:
    raise ValueError(f"No valid images found in {custom_image_folder}. Please check the folder path or image naming conventions.")

# confirm = input("Confirm if these are correct action-image pairs (Press Enter to confirm, or type 'n' to cancel): ")
# if confirm.strip().lower() == 'n':
#     raise ValueError("User cancelled confirmation.")


print("\n=== Loading test images ===")
# === Load test images from test_set folder (no actions) ===
test_image_paths = sorted(glob.glob(os.path.join(test_folder, "*.JPG")) + 
                        glob.glob(os.path.join(test_folder, "*.jpg")))

print(f"\nFound {len(test_image_paths)} test images in '{test_folder}', all good!!")
if not test_image_paths:
    # print a warning if no test images found
    print(f"Warning: No test images found in '{test_folder}'. Please check the folder path or image naming conventions.")
    input("Press Enter to continue...")


# === Device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU if available


H = W = image_size  # height and width of image

# === Define start/goal positions for 2 start-goal pairs ===
img_center = (W // 2, H // 2)  # center of image grid

if same_goal_positions:
    # If goals are same, both goal positions point to the center
    goal_positions = [(img_center[0], img_center[1]), (img_center[0], img_center[1])]
else:
    # If different, offset goals left and right from center
    goal_positions = [(img_center[0] + W // 4, img_center[1]), (img_center[0] - W // 4, img_center[1])]

# Derive corresponding start positions symmetrically around goals
start_positions = [(goal_positions[0][0] - W // 4, goal_positions[0][1]),
                   (goal_positions[1][0] + W // 4, goal_positions[1][1])]

# Combine into (start, goal) pairs
start_goal_pairs = []
for idx in range(len(start_positions)):
    start = start_positions[idx]
    goal = goal_positions[idx]
    start_goal_pairs.append((start, goal))
    print (f"Start: {start}, Goal: {goal}, start-goal is {goal[0] - start[0]}, {goal[1] - start[1]}")
samples_per_pair = N // len(start_goal_pairs)  # distribute samples equally


images = []  # to hold input image tensors
actions = []  # to hold target action vectors

# === Load and transform image utility ===
def load_and_prepare_image(img_path, size=(64, 64), augment=False):
    img = Image.open(img_path).convert('RGB')
    transform_list = [transforms.Resize(size)]

    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),  # left/right orientation
            transforms.RandomAffine(
                degrees=15,            # small rotation to simulate slight viewpoint tilt
                translate=(0.1, 0.1),  # up to 10% translation in x/y
                scale=None,
                shear=None
            )
        ])

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    img_tensor = transform(img)
    img_tensor = img_tensor[:3]
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor


# # === Construct final dataset and DataLoader ===

image_tensors = []
action_tensors = []
image_pose_tensors = []  # to hold 3rd view camera poses if needed


# Prepare full dataset from auto-parsed inputs
# images, actions, view_poses = prepare_dataset(custom_image_inputs, N)

if include_pose_in_input:
    images, actions, view_poses = prepare_dataset(custom_image_inputs, N)
    dataset = TensorDataset(images, actions, view_poses)
else:
    images, actions = prepare_dataset(custom_image_inputs, N)
    dataset = TensorDataset(images, actions)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Shuffle
perm = torch.randperm(N)
images = images[perm]
actions = actions[perm]
if include_pose_in_input:
    view_poses = view_poses[perm]

if include_pose_in_input:
    dataset = TensorDataset(images, actions, view_poses)
else:
    dataset = TensorDataset(images, actions)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




# === Define the model architecture ===
class DiffusionPolicy(nn.Module):
    def __init__(self, image_channels, embed_dim, pose_dim, include_pose_in_input):
        super(DiffusionPolicy, self).__init__()
        self.include_pose_in_input = include_pose_in_input
        self.embed_dim = embed_dim

        # Image encoder
        self.conv1 = nn.Conv2d(image_channels, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2, padding=2)
        self.conv_out = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_img = nn.Linear(32, embed_dim)

        # Time embedding
        self.time_dim = embed_dim
        self.fc_t1 = nn.Linear(embed_dim, embed_dim)
        self.fc_t2 = nn.Linear(embed_dim, embed_dim)

        # Action embedding
        self.fc_action = nn.Linear(2, embed_dim)

        # Pose FiLM layers
        if self.include_pose_in_input:
            self.pose_to_scale1 = nn.Linear(pose_dim, embed_dim)
            self.pose_to_shift1 = nn.Linear(pose_dim, embed_dim)
            self.pose_to_scale2 = nn.Linear(pose_dim, embed_dim)
            self.pose_to_shift2 = nn.Linear(pose_dim, embed_dim)

        # Main MLP layers
        self.fc1 = nn.Linear(embed_dim * 3, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, 2)

    def forward(self, x_t, image, t, pose_2d=None):
        B = image.size(0)

        # Image encoding
        h = torch.relu(self.conv1(image))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        h = self.conv_out(h).view(B, -1)
        img_feat = torch.relu(self.fc_img(h))  # [B, D]

        # Time embedding
        t = t.float().to(image.device)
        half_dim = self.time_dim // 2
        inv_freq = torch.exp(torch.arange(half_dim, device=image.device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        t_emb = t.unsqueeze(1) * inv_freq.unsqueeze(0)
        t_emb = torch.cat([t_emb.sin(), t_emb.cos()], dim=1)
        t_feat = torch.relu(self.fc_t1(t_emb))
        t_feat = torch.relu(self.fc_t2(t_feat))  # [B, D]

        # Action embedding
        a_feat = torch.relu(self.fc_action(x_t))  # [B, D]

        # Concatenate
        combined = torch.cat([img_feat, t_feat, a_feat], dim=1)  # [B, 3*D]
        h = self.fc1(combined)  # [B, D]

        if self.include_pose_in_input:
            assert pose_2d is not None, "Pose must be provided if include_pose_in_input is True"
            scale1 = self.pose_to_scale1(pose_2d)
            shift1 = self.pose_to_shift1(pose_2d)
            h = h * scale1 + shift1

        h = torch.relu(h)
        h = self.fc2(h)

        if self.include_pose_in_input:
            scale2 = self.pose_to_scale2(pose_2d)
            shift2 = self.pose_to_shift2(pose_2d)
            h = h * scale2 + shift2

        h = torch.relu(h)
        eps_pred = self.fc_out(h)
        return eps_pred

# Instantiate model, optimizer, scheduler
model = DiffusionPolicy(image_channels=3, embed_dim=embed_dimension_feature_vector, pose_dim=2, include_pose_in_input=include_pose_in_input).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

# === Define diffusion schedule ===
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1.0 - betas
alpha_cum = torch.cumprod(alphas, dim=0).to(device)
alpha_cum = torch.cat([torch.tensor([1.0], device=device), alpha_cum])  # pad for t=0

# === Training Loop ===
best_loss = float('inf')
loss_history = []
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save_dir = f"viz_outputs/domain_generalization_diffusion_N_{N}_epoch_{num_epochs}_{timestamp_str}"
if include_pose_in_input:
    print("Including 3rd view camera pose in input, so saving to a different directory")
    if run_inference_only:
        # check and assert with_viewpoint should be in the original model (inference_model_path)
        if "with_viewpoint_condition" not in inference_model_path or "without_viewpoint_condition" in inference_model_path:
            raise ValueError("Inference model path does not contain 'with_viewpoint_condition', but include_pose_in_input is True.")
        else:
            # get the inference_model_path, the save_dir should be the same as inference_model_path's folder
            temp_parent_dir = os.path.dirname(inference_model_path)
            # add inference_only as prefix, but after viz_outputs/
            save_dir = os.path.join(temp_parent_dir, f"inference_only_{os.path.basename(inference_model_path)}")
            print(f"Saving inference only model to: {save_dir}")
        # save_dir = f"viz_outputs/inference_only_domain_generalization_diffusion_N_{N}_epoch_{num_epochs}_with_viewpoint_condition"
    else:
        save_dir = f"viz_outputs/domain_generalization_diffusion_N_{N}_epoch_{num_epochs}_with_viewpoint_condition"
else:
    if run_inference_only:
        # save_dir = f"viz_outputs/inference_only_domain_generalization_diffusion_N_{N}_epoch_{num_epochs}_without_viewpoint_condition"
        # check and assert with_viewpoint should NOT be in the original model (inference_model_path)
        if "with_viewpoint_condition" in inference_model_path or "without_viewpoint_condition" not in inference_model_path:
            raise ValueError("Inference model path contains 'with_viewpoint_condition', but include_pose_in_input is False.")
        else:
            # get the inference_model_path, the save_dir should be the same as inference_model_path's folder
            temp_parent_dir = os.path.dirname(inference_model_path)
            # add inference_only as prefix, but after viz_outputs/
            save_dir = os.path.join(temp_parent_dir, f"inference_only_{os.path.basename(inference_model_path)}")
            print(f"Saving inference only model to: {save_dir}")
    else:
        save_dir = f"viz_outputs/domain_generalization_diffusion_N_{N}_epoch_{num_epochs}_without_viewpoint_condition"

if run_inference_only:
    if use_data_augmentation:
        save_dir += "_with_data_augmentation"
    else:
        save_dir += "_without_data_augmentation"

    save_dir += "_embed_dim_" + str(embed_dimension_feature_vector)

    save_dir += f"_{timestamp_str}" 

train_vis_dir = os.path.join(save_dir, "train_set")
test_vis_dir = os.path.join(save_dir, "test_set")
os.makedirs(train_vis_dir, exist_ok=True)
os.makedirs(test_vis_dir, exist_ok=True)
# os.makedirs(save_dir, exist_ok=True)
if not run_inference_only:
    model_path = os.path.join(save_dir, "best_model.pt")
    print(f"Saving model checkpoints to: {model_path}")
else:
    model_path = os.path.join(inference_model_path, "best_model.pt")
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Running inference only")
    print(f"Running inference only")
    print(f"Running inference only")
    print(f"Running inference only")
    print(f"Running inference only")
    print(f"Running inference only")
    print(f"Running inference only")
    print(f"Running inference only")
    print(f"Running inference only, loading model from: {model_path}")
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    

if not run_inference_only:
    print("Training model...\n")
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        model.train()
        for batch in train_loader:
            if include_pose_in_input:
                batch_images, batch_actions, batch_poses = batch
                batch_poses = batch_poses.to(device)
            else:
                batch_images, batch_actions = batch
                batch_poses = None

            batch_images = batch_images.to(device)
            batch_actions = batch_actions.to(device)
            batch_size_curr = batch_images.size(0)
            t = torch.randint(1, T + 1, (batch_size_curr,), device=device)
            alpha_bar_t = alpha_cum[t].view(-1, 1)
            eps = torch.randn(batch_size_curr, 2, device=device)
            x0 = batch_actions
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
            eps_pred = model(x_t, batch_images, t, batch_poses)
            weight = torch.tensor([1.0, 1.0], device=device)
            loss = ((eps_pred - eps) * weight).pow(2).mean()  # MSE loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size_curr
        avg_loss = epoch_loss / len(dataset)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model (loss {best_loss:.6f})")
        scheduler.step(avg_loss)
        loss_history.append(avg_loss)

    # === Plot and save training loss curve ===
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss_curve.png"))
    plt.close()
    gc.collect()
else:
    print("Skipping training, only running inference on best model...")

# === Load best model for inference ===
print("\nLoading best model for inference...")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

if include_pose_in_input:
    print("FiLM layer norms:")
    print("  Scale1 weight norm:", model.pose_to_scale1.weight.norm().item())
    print("  Shift1 weight norm:", model.pose_to_shift1.weight.norm().item())
    print("  Scale2 weight norm:", model.pose_to_scale2.weight.norm().item())
    print("  Shift2 weight norm:", model.pose_to_shift2.weight.norm().item())

# === Inference and Visualization ===



print(f"Only visualizing every {gif_skip_interval}th frame for GIFs to reduce size.")

# Process training set images
if not run_test_set_only:
    for img_path, true_action, cur_view_deg in custom_image_inputs:
        print(f"Processing custom image: {img_path}")
        test_img = load_and_prepare_image(img_path, size=(W, H), augment=False)
        img_tensor = test_img.to(device).unsqueeze(0)
        if include_pose_in_input:
            # Use the view degree as the yaw of the 3rd view camera pose
            # third_view_cam_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, cur_view_deg * np.pi / 180.0], device=device).unsqueeze(0)
            yaw_rad = cur_view_deg * np.pi / 180.0
            third_view_cam_pose = torch.tensor([np.sin(yaw_rad), np.cos(yaw_rad)], dtype=torch.float32, device=device).unsqueeze(0)
            print(f"Using custom view degree {cur_view_deg} for pose: {third_view_cam_pose}")
        else:
            third_view_cam_pose = None

        dx_true, dy_true = true_action
        img_np = ((test_img + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()

        # Save raw image
        raw_name = f"custom_raw_input_{os.path.basename(img_path)}"
        plt.imsave(os.path.join(train_vis_dir, raw_name), img_np)

        # Annotated input
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)
        ax.arrow(W//2, H//2, dx_true * (W / 2), dy_true * (H / 2),
                    head_width=1.5, head_length=3, color='gray', linestyle='dashed',
                    linewidth=2, label='True Action')
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_title(f"Custom Input: {os.path.basename(img_path)}")
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(train_vis_dir, f"custom_input_{os.path.basename(img_path)}.png"))
        plt.close()
        gc.collect()

        # Generate sample trajectories
        all_trajectories = []
        for _ in range(num_inference_samples):
            x = torch.randn(1, 2, device=device)
            traj = [x.squeeze(0).detach().cpu().numpy().copy()]
            for t_step in range(T, 0, -1):
                t_tensor = torch.tensor([t_step], device=device)
                eps_pred = model(x, img_tensor, t_tensor, third_view_cam_pose)
                alpha_t = alphas[t_step - 1]
                alpha_bar_t = alpha_cum[t_step]
                x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_t)
                traj.append(x.squeeze(0).detach().cpu().numpy().copy())
            all_trajectories.append(traj)

        # Generate animated overlay of trajectories
        frames = []
        for t_idx in range(T + 1):
            if t_idx % gif_skip_interval != 0 and t_idx != T:
                continue  # skip frames for GIF
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np)
            ax.arrow(W//2, H//2, dx_true * (W / 6), dy_true * (H / 6),
                    head_width=1.5, head_length=3, color='green', linewidth=2)
            ax.text(2, 10, "Green = Ground Truth", fontsize=10, color='green')

            for traj in all_trajectories:
                dx = traj[t_idx][0] * (W / 6)
                dy = traj[t_idx][1] * (H / 6)
                ax.arrow(W//2, H//2, dx, dy, head_width=0.5, head_length=1.5,
                            color='red', alpha=0.1, linewidth=1)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_title(f"{os.path.basename(img_path)} | Step {t_idx}")
            ax.grid(True)
            fig.canvas.draw()
            frame = np.array(fig.canvas.buffer_rgba())
            frame = frame[:, :, :3]  # keep RGB, discard alpha
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close()
            gc.collect()

        gif_path = os.path.join(train_vis_dir, f"custom_traj_{os.path.basename(img_path).split('.')[0]}.gif")
        imageio.mimsave(gif_path, frames, duration=0.05)
        print(f"Saved: {gif_path}")



# === TEST ALL .JPG IMAGES IN "test_set" FOLDER ===
for img_path in test_image_paths:
    pose_dim = 2
    print(f"Testing image: {img_path}")
    test_img = load_and_prepare_image(img_path, size=(W, H), augment=False)
    img_tensor = test_img.to(device).unsqueeze(0)
    img_np = ((test_img + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
    # third view camera pose
    if include_pose_in_input:
        # extract the image filename to check if it contains "view"
        img_filename = os.path.basename(img_path).lower()
        if "view" in img_filename:
            # Extract view number from filename
            print(f"Extracting view number from filename: {img_path}")
            view_num = int(img_filename.split('view')[1].split('.')[0])
            cur_view_deg = view_num_to_deg(view_num, training_view_nums)
            yaw_rad = cur_view_deg * np.pi / 180.0
            # third_view_cam_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, cur_view_deg * np.pi / 180.0], device=device).unsqueeze(0)
            third_view_cam_pose = torch.tensor([np.sin(yaw_rad), np.cos(yaw_rad)], dtype=torch.float32, device=device).unsqueeze(0)
            print(f"Current viewpoint: {cur_view_deg} degrees")
        else:
            # third_view_cam_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0)
            third_view_cam_pose = torch.zeros((1, pose_dim), device=device)  # dummy pose
            print("WARNING: Using dummy third view camera pose for testing, since no pose provided in test images!")

    # Save raw image
    raw_name = f"test_raw_input_{os.path.basename(img_path)}"
    plt.imsave(os.path.join(test_vis_dir, raw_name), img_np)

    # Run diffusion sampling
    all_trajectories = []
    for _ in range(num_inference_samples):
        x = torch.randn(1, 2, device=device)
        traj = [x.squeeze(0).detach().cpu().numpy().copy()]
        for t_step in range(T, 0, -1):
            t_tensor = torch.tensor([t_step], device=device)
            if include_pose_in_input:
                # third_view_cam_pose is a tensor of shape [1, 6]
                eps_pred = model(x, img_tensor, t_tensor, third_view_cam_pose)
            else:
                # print("Not including 3rd view camera pose in input, so third_view_cam_pose is None")
                eps_pred = model(x, img_tensor, t_tensor, None)
            alpha_t = alphas[t_step - 1]
            alpha_bar_t = alpha_cum[t_step]
            x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_t)
            traj.append(x.squeeze(0).detach().cpu().numpy().copy())
        all_trajectories.append(traj)

    # Generate animated overlay
    frames = []
    img_h, img_w = img_np.shape[0], img_np.shape[1]
    for t_idx in range(T + 1):
        if t_idx % gif_skip_interval != 0 and t_idx != T:
            continue  # skip frames for GIF
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)
        for traj in all_trajectories:
            dx = traj[t_idx][0] * (img_w / 6)
            dy = traj[t_idx][1] * (img_h / 6)
            ax.arrow(img_w//2, img_h//2, dx, dy, head_width=0.5, head_length=1.5,
                    color='red', alpha=0.1, linewidth=1)
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_title(f"{os.path.basename(img_path)} | Step {t_idx}")
        ax.grid(True)
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())
        frame = frame[:, :, :3]  # keep RGB, discard alpha
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
        gc.collect()

    gif_path = os.path.join(test_vis_dir, f"test_traj_{os.path.basename(img_path).split('.')[0]}.gif")
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"Saved: {gif_path}")

########################################### NEW NETWORK ARCHITECTURE DIAGRAM ###########################################
########################################### NEW NETWORK ARCHITECTURE DIAGRAM ###########################################
#                              +------------------------------+
#                              |  Viewpoint Encoding [sinθ,   |
#                              |        cosθ] ∈ ℝ²           |
#                              +------------------------------+
#                                       |         \
#                 (Linear: 2 → 64 for scale1)    (Linear: 2 → 64 for shift1)
#                                       |           |
#                               +--------------+   +--------------+
#                               |  FiLM Scale1 |   |  FiLM Shift1 |
#                               +--------------+   +--------------+

#                  (Linear: 2 → 64 for scale2)    (Linear: 2 → 64 for shift2)
#                          |                           |
#                +-------------------+        +--------------------+
#                |   FiLM Scale2     |        |   FiLM Shift2      |
#                +-------------------+        +--------------------+


#   +--------------------+     +--------------------+     +------------------------+
#   |  Image [B,3,H,W]   |     |  Noisy Action x_t  |     | Diffusion Timestep t  |
#   +--------------------+     +--------------------+     +------------------------+
#            |                          |                            |
# (3 conv layers + avg pool + FC)   (Linear 2 → 64)        (Sin/cos embedding + MLP)
#            |                          |                            |
#            v                          v                            v
#   [Image Embedding]        [Action Embedding]            [Time Embedding]
#         ∈ ℝ⁶⁴                     ∈ ℝ⁶⁴                          ∈ ℝ⁶⁴

#               \                |              /
#                \               |             /
#                 +-------------+-------------+
#                               |
#                       Concatenate ∈ ℝ¹⁹²
#                               |
#                      Fully Connected (FC1)
#                               |
#                       Apply FiLM Scale1/Shift1
#                               |
#                             ReLU
#                               |
#                      Fully Connected (FC2)
#                               |
#                       Apply FiLM Scale2/Shift2
#                               |
#                             ReLU
#                               |
#                     Fully Connected (FC3 → ℝ²)
#                               |
#                     Predicted Noise ε ∈ ℝ²














########################################### OLD NETWORK ARCHITECTURE DIAGRAM ###########################################
########################################### OLD NETWORK ARCHITECTURE DIAGRAM ###########################################
    #                             +------------------+
    #                             |  6DOF Pose [6]   |
    #                             +------------------+
    #                                      |
    #                              (Linear: 6 -> 64)
    #                                      |
    #                               (ReLU activation)
    #                                      |
    #                             +------------------+
    #                             |  Pose Embedding  |
    #                             |   [Batch, 64]    |
    #                             +------------------S+




    #   +--------------------+        +------------------+        +--------------------+
    #   |  Image [B,3,H,W]   |        |  Noisy Action    |        |  Diffusion Timestep|
    #   +--------------------+        |   x_t [B,2]      |        |   t [B]            |
    #             |                   +------------------+        +--------------------+
    #     (3 conv layers +            |                         (Sinusoidal encoding +
    #   adaptive avg pool +           |                        2-layer MLP to [B,64])
    #     FC: 32->64)                 |                             |
    #             |                   |                             |
    #     (ReLU activation)           |                             |
    #             |                   |                             |
    #   +-------------------+     (Linear 2->64)            +---------------------+
    #   | Image Embedding   |----- (ReLU) ---------+         |  Time Embedding     |
    #   |   [Batch, 64]     |                     |         |   [Batch, 64]       |
    #   +-------------------+                     |         +---------------------+
    #                                             |
    #   +-------------------+   +-----------------+   +-------------------+   +-------------------+
    #   | Action Embedding  |   |  Pose Embedding |   |  Time Embedding   |   |  Image Embedding  |
    #   +-------------------+   +-----------------+   +-------------------+   +-------------------+
    #           \                   /                      /                     /
    #            \                 /                      /                     /
    #             +-------------------------------------------------------------+
    #             |                        Concat                             |
    #             +--------------------------+----------------------------------+
    #                                        |
    #                               (FC: 64 x 4 -> 64, ReLU)
    #                                        |
    #                               (FC: 64 -> 64, ReLU)
    #                                        |
    #                               (FC: 64 -> 2)  <-- Output: noise prediction
    #                                        |
    #                                 [Batch, 2] (ε_pred)
