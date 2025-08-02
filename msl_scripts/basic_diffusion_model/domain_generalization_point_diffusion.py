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



"""
MIT License
...
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import time
import imageio


# Set to True to use diffusion, or False to use plain MLP
USE_DIFFUSION = True


# Configuration
shape_size = 2.0
high_data_samples = 100000
num_epochs = 100
num_timesteps = 100
num_inference_samples = 500
hidden_size = 1024
hidden_layers = 10
learning_rate = 1e-3
batch_size = 4096
plot_limit = shape_size * 1.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("bidirectional_outputs", f"run_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

def sample_bidirectional_trajectories(n_samples, goal=np.array([0.0, 0.0]), distance=2.0):
    half = n_samples // 2
    remainder = n_samples % 2
    left = np.array([[goal[0] - distance, goal[1]]], dtype=np.float32)
    right = np.array([[goal[0] + distance, goal[1]]], dtype=np.float32)
    starts = np.repeat(np.vstack([left, right]), [half, half + remainder], axis=0)
    goals = np.repeat(np.array([goal], dtype=np.float32), n_samples, axis=0)
    return goals, starts

class DiffusionMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=2, hidden_size=128, hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ActionMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_size=128, hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

T = num_timesteps
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def plot_loss_curve(losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training loss plot to: {save_path}")
    
def train_model(model, points, epochs, lr, batch_size, save_plot_path=None, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    model.train()
    best_loss = float('inf')
    best_state_dict = None
    loss_history = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        perm = torch.randperm(points.size(0))
        x0 = points[perm]
        epoch_loss = 0.0

        for i in range(0, x0.size(0), batch_size):
            batch_x0 = x0[i:i+batch_size]
            t = torch.randint(1, T + 1, (batch_x0.size(0),), device=device)
            alpha_bar_t = alpha_bars[t - 1].unsqueeze(1)
            epsilon = torch.randn_like(batch_x0)
            x_t = torch.sqrt(alpha_bar_t) * batch_x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
            t_norm = (t - 1).unsqueeze(1).float() / (T - 1)
            model_input = torch.cat([x_t, t_norm], dim=1)
            pred = model(model_input)
            loss = loss_fn(pred, epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x0.size(0)

        scheduler.step()
        epoch_loss /= points.size(0)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1:03d}/{epochs}: Loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    if save_plot_path is not None:
        plot_loss_curve(loss_history, save_path=save_plot_path)
    return model


def train_action_model(model, start_points, goal_points, epochs, lr, batch_size, save_plot_path=None, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    loss_history = []

    inputs = torch.from_numpy(start_points).float().to(device)
    targets = torch.from_numpy(goal_points).float().to(device)

    best_loss = float('inf')
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        perm = torch.randperm(inputs.size(0))
        epoch_loss = 0.0

        for i in range(0, inputs.size(0), batch_size):
            x_batch = inputs[perm[i:i+batch_size]]
            y_batch = targets[perm[i:i+batch_size]]
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= inputs.size(0)
        loss_history.append(epoch_loss)
        print(f"[Plain MLP] Epoch {epoch+1:03d}/{epochs}: Loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[Plain MLP] Early stopping triggered at epoch {epoch+1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    if save_plot_path is not None:
        plot_loss_curve(loss_history, save_path=save_plot_path)
    return model

def sample_trajectories(model, initial_noise):
    x_t = initial_noise.to(device)
    traj = [x_t.cpu().numpy()]
    for t in reversed(range(1, T + 1)):
        t_tensor = torch.full((x_t.size(0),), t, device=device, dtype=torch.long)
        t_norm = (t_tensor - 1).unsqueeze(1).float() / (T - 1)
        with torch.no_grad():
            pred_eps = model(torch.cat([x_t, t_norm], dim=1))
        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]
        alpha_bar_prev = alpha_bars[t - 2] if t > 1 else torch.tensor(1.0, device=device)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
        mu = coef1.unsqueeze(0) * x0_pred + coef2.unsqueeze(0) * x_t
        posterior_variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
        sigma = torch.sqrt(posterior_variance)
        if t > 1:
            z = torch.randn_like(x_t)
            x_t = mu + sigma * z
        else:
            x_t = mu
        traj.append(x_t.cpu().numpy())
    return np.stack(traj, axis=1)

def plot_trajectories(traj, ground_truth, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    starts, ends = traj[:, 0], traj[:, -1]
    segments = np.stack([starts, ends], axis=1)
    lc = LineCollection(segments, colors='cyan', linewidths=0.3, alpha=0.8)
    ax.add_collection(lc)
    ax.scatter(starts[:, 0], starts[:, 1], c='gray', s=20, alpha=0.2, label='Initial')
    ax.scatter(ends[:, 0], ends[:, 1], c='blue', s=10, alpha=0.4, label='Predicted')
    ax.scatter(ground_truth[:, 0], ground_truth[:, 1], c='orange', s=30, label='Ground Truth')
    ax.scatter([0], [0], c='red', marker='*', s=100, label='Goal')
    ax.axis('equal')
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_title(title or "Diffusion Trajectories")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.close(fig)

def interpolate_color(c1, c2, alpha):
    return tuple((1 - alpha) * a + alpha * b for a, b in zip(c1, c2))

def save_trajectory_video(trajectories, shape_type="custom", save_path="diffusion.mp4", fps=15, tail_length=20, ground_truth_points=None, start_points=None):
    from matplotlib.colors import to_rgb
    num_samples, num_steps, _ = trajectories.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    frames = []
    gray_rgb = to_rgb("gray")
    blue_rgb = to_rgb("blue")
    trail_color = (0.3, 0.3, 0.3)
    min_size = 1
    max_size = 40
    subset = min(num_samples, 300)
    marker_size_cur = 30 if ground_truth_points is not None and ground_truth_points.shape[0] < high_data_samples else 7
    last_rendered_frame = None

    for t in range(1, num_steps):
        ax.clear()
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.axis('off')
        ax.set_title(f"{shape_type.upper()} Diffusion Step {t}/{num_steps - 1}", fontsize=14)
        segments = []
        segment_colors = []
        segment_widths = []
        overall_alpha_scale = t / (num_steps - 1)
        for i in range(subset):
            for dt in range(tail_length):
                t1 = t - dt - 1
                t2 = t1 + 1
                if t1 < 0 or t2 >= num_steps:
                    continue
                p1 = trajectories[i, t1]
                p2 = trajectories[i, t2]
                fade = dt / tail_length
                base_alpha = 0.1 + 0.2 * (1 - fade)
                alpha = base_alpha * overall_alpha_scale
                width = 0.6 + 0.5 * (1 - fade)
                segments.append([p1, p2])
                segment_colors.append((*trail_color, alpha))
                segment_widths.append(width)
        if segments:
            lc = LineCollection(segments, colors=segment_colors, linewidths=segment_widths, zorder=1)
            ax.add_collection(lc)
        alpha = t / (num_steps - 1)
        sizes = min_size + alpha * (max_size - min_size)
        colors = [interpolate_color(gray_rgb, blue_rgb, alpha) for _ in range(num_samples)]
        ax.scatter(trajectories[:, t, 0], trajectories[:, t, 1], s=sizes, c=colors, edgecolors='white', linewidths=0.2, alpha=0.25, zorder=2)
        if ground_truth_points is not None:
            ax.scatter(ground_truth_points[:, 0], ground_truth_points[:, 1], c='orange', s=marker_size_cur, linewidths=0.1, alpha=1.0, zorder=100)
        if start_points is not None:
            ax.scatter(start_points[:, 0], start_points[:, 1], c='gray', s=marker_size_cur, alpha=0.6, zorder=90)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        last_rendered_frame = frame

    frames += [last_rendered_frame] * 10
    imageio.mimsave(save_path, frames, fps=fps)
    plt.close(fig)
    print(f"Saved diffusion video to: {save_path}")

# ============================ MAIN ============================
if __name__ == "__main__":
    goals_np, starts_np = sample_bidirectional_trajectories(high_data_samples, distance=shape_size)
    FIXED_START_POINTS = starts_np.astype(np.float32)
    FIXED_GOAL_POINTS = goals_np.astype(np.float32)
    indices = np.random.choice(FIXED_START_POINTS.shape[0], num_inference_samples, replace=False)
    noise_infer = torch.from_numpy(FIXED_START_POINTS[indices]).to(device)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    loss_plot_path = os.path.join(save_dir, f"loss_{timestamp}.png")

    if USE_DIFFUSION:
        model = DiffusionMLP(input_dim=3, output_dim=2, hidden_size=hidden_size, hidden_layers=hidden_layers).to(device)
        model = train_model(model, torch.from_numpy(FIXED_GOAL_POINTS).to(device), num_epochs, learning_rate, batch_size, save_plot_path=loss_plot_path, patience=num_epochs // 5)
        traj = sample_trajectories(model, noise_infer)
    else:
        model = ActionMLP(input_dim=2, output_dim=2, hidden_size=hidden_size, hidden_layers=hidden_layers).to(device)
        model = train_action_model(model, FIXED_START_POINTS, FIXED_GOAL_POINTS, num_epochs, learning_rate, batch_size, save_plot_path=loss_plot_path, patience=num_epochs // 5)
        with torch.no_grad():
            predicted = model(torch.from_numpy(FIXED_START_POINTS[indices]).float().to(device)).cpu().numpy()
        traj = np.stack([FIXED_START_POINTS[indices], predicted], axis=1)

    fig_path = os.path.join(save_dir, f"trajectories_{timestamp}.png")
    video_path = os.path.join(save_dir, f"video_{timestamp}.mp4")
    plot_trajectories(traj, FIXED_GOAL_POINTS[indices], save_path=fig_path, title="Bidirectional Model (Diffusion)" if USE_DIFFUSION else "Bidirectional Model (Plain MLP)")
    save_trajectory_video(traj, shape_type="bidirectional", save_path=video_path, ground_truth_points=FIXED_GOAL_POINTS[indices], start_points=FIXED_START_POINTS[indices])
