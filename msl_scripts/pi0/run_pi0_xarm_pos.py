import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from xarm.wrapper import XArmAPI
import time
import sys
sys.path.append('../../xArm-Python-SDK')

# -------------------------
# Initialize Robot
arm = XArmAPI('192.168.1.219')
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)
arm.set_gripper_mode(0)
arm.set_gripper_speed(2000)
arm.set_gripper_enable(True)
arm.set_pause_time(0.0)
print("[INFO] Robot initialized.")

# -------------------------
# Load Policy
print("[INFO] Loading Pi0 policy...")
policy = PI0Policy.from_pretrained("lerobot/pi0")
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
print("[INFO] Policy and tokenizer loaded.")

goal = ["pick up the cup"]
print(f"[INFO] Goal: {goal[0]}")

# -------------------------
# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# -------------------------
# RealSense Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# -------------------------
# Pi0 Direct Inference
def select_action_direct(policy, image_tensor, goal_text):
    model = policy.model
    device = next(model.parameters()).device
    tokens = tokenizer(goal_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attn_mask = tokens["attention_mask"].to(device)
    seq_len = min(input_ids.shape[1], attn_mask.shape[1])
    input_ids = input_ids[:, :seq_len]
    attn_mask = attn_mask[:, :seq_len]

    vision = image_tensor.to(dtype=torch.float32, device=device).unsqueeze(0)
    img_masks = torch.ones((1, 1, 1, 1), dtype=torch.bool, device=device)
    state = torch.zeros(1, policy.config.max_state_dim, device=device)
    actions = torch.zeros(1, policy.config.max_action_dim, device=device)

    with torch.no_grad():
        action_pred = model.forward(vision, img_masks, input_ids, attn_mask, state, actions)
    return action_pred

# -------------------------
# Command Robot with First 6 Joint Angles
def command_robot_with_action(action_tensor):
    action = action_tensor[0][:6].detach().cpu().numpy()  # First 6 DOF
    print(f"[ACTION] First 6 joint angles (radians): {np.round(action, 4)}")
    
    ret = arm.set_position(
        joint=action.tolist(),
        speed=30,
        wait=True,
        is_radian=True
    )
    if ret != 0:
        print(f"[WARN] set_position failed with code {ret}")
    else:
        print("[INFO] Robot moved to predicted joint positions.")

# -------------------------
# Main Inference + Control Loop
try:
    pipeline.start(config)
    print("[INFO] Streaming started.")
    device = next(policy.parameters()).device

    frame_count = 0
    while frame_count < 10:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("[WARN] No frame received.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        image_pil = Image.fromarray(color_image).convert("RGB")
        image_tensor = preprocess(image_pil).to(device)

        print(f"[INFO] Frame {frame_count+1}: running policy inference...")
        action = select_action_direct(policy, image_tensor, goal[0])
        command_robot_with_action(action)

        frame_count += 1
        time.sleep(0.5)  # Small delay between actions

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    print("[INFO] Stopping pipeline and cleaning up.")
    pipeline.stop()
    arm.disconnect()
