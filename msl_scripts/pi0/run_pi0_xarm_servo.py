import torch
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from transformers import AutoTokenizer
from xarm.wrapper import XArmAPI
import time

import sys
sys.path.append('../../xArm-MSL-ROS/xArm-Python-SDK')

# ----------------------------
# Configurations
# ----------------------------
XARM_IP = '192.168.1.219'
USE_RADIANS = False
MOVE_SPEED = 0.5  # rad/s

# ----------------------------
# Initialize robot
# ----------------------------
print("[INFO] Initializing xArm...")
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)
print("[INFO] Motion enabled")
arm.set_mode(0)
print("[INFO] Mode set to 0")
arm.set_state(0)
print("[INFO] State set to 0")

# Initialize gripper
print("[INFO] Initializing gripper...")
arm.set_gripper_mode(0)
arm.set_gripper_speed(2000)
arm.set_gripper_enable(True)
print("[INFO] Opening gripper to full open position...")
arm.set_gripper_position(850, wait=True)

# ----------------------------
# Load Policy and Tokenizer
# ----------------------------
print("[INFO] Loading PI0Policy model and tokenizer...")
policy = PI0Policy.from_pretrained("lerobot/pi0")
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
print("[INFO] Model and tokenizer loaded.")

# ----------------------------
# Image Preprocessing
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

goal_text = "pick up the cup"

# ----------------------------
# RealSense Setup
# ----------------------------
print("[INFO] Starting RealSense pipeline...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("[INFO] RealSense pipeline started.")

def run_pi0_inference(policy, image_tensor, goal):
    model = policy.model
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    tokens = tokenizer(goal, return_tensors="pt", padding=True, truncation=True).to(device)
    img_masks = torch.ones((1, 1, 1, 1), dtype=torch.bool, device=device)

    # Get current joint angles from xArm (radians or degrees based on your config)
    ret_code, current_angles = arm.get_servo_angle()
    if ret_code < 0:
        print(f"[ERROR] Failed to get current joint angles, code: {ret_code}")
        return None
    current_angles = current_angles[:6]  # Only take first 6 angles for the arm
    
    if ret_code != 0:
        print(f"[WARN] Failed to get current joint angles, code: {ret_code}")
        current_angles = [0]*6
    else:
        print(f"[INFO] Current joint angles: {current_angles}")

    # Convert to torch tensor and send to device
    current_angles_tensor = torch.tensor(current_angles, dtype=torch.float32, device=device).unsqueeze(0)

    # Initialize state vector, fill first 6 dims with current joint angles,
    # zero-pad remaining dims if state dim is larger
    state = torch.zeros((1, policy.config.max_state_dim), device=device)
    state[:, :6] = current_angles_tensor

    # Initialize actions to zeros
    actions = torch.zeros((1, policy.config.max_action_dim), device=device)

    print("[INFO] Running inference on PI0 policy model...")
    with torch.no_grad():
        out = model(image_tensor, img_masks, tokens["input_ids"], tokens["attention_mask"], state, actions)
    print("[INFO] Inference done.")
    return out

# ----------------------------
# Main Loop
# ----------------------------
try:
    print("[INFO] Starting main inference loop...")
    for i in range(10):
        print(f"[INFO] Frame iteration {i+1}/10")
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("[WARN] No frame received.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        image_pil = Image.fromarray(color_image).convert("RGB")
        image_tensor = preprocess(image_pil)

        print("[INFO] Preprocessed current frame for model input.")

        action_tensor = run_pi0_inference(policy, image_tensor, goal_text)
        action_np = action_tensor.cpu().numpy().flatten()
        print(f"[DEBUG] Full action tensor output: {action_np}")
        
        # this will be a list of future actions, not just one
        # check if the output is a batch of actions
        if action_np.ndim == 2:
            print("[INFO] Action tensor has batch dimension, executing one by one.")
            for action in action_np:
                joint_angles = action[:6]
                # print in 2 decimal places in degrees
                joint_angles = np.round(np.degrees(joint_angles), 2) if not USE_RADIANS else np.round(joint_angles, 2)

                # print diff between current and target angles
                current_angles = arm.get_servo_angle(is_radian=USE_RADIANS)[1][:6]
                angle_diff = np.array(joint_angles) - np.array(current_angles)
                # print in 2 decimal places in degrees
                angle_diff = np.round(np.degrees(angle_diff), 2) if not USE_RADIANS else np.round(angle_diff, 2)
                angle_diff_str = ', '.join(f"{x:.2f}" for x in angle_diff)
                # if angle diff is more than 50 degrees, print a warning and continue
                if np.any(np.abs(angle_diff) > 50):
                    print("[WARN] Large angle difference detected,  skipping action to prevent large movements.")
                    continue
                print(f"[DEBUG] Angle difference from current: {angle_diff_str} degrees")
                arm.set_servo_angle(angle=joint_angles.tolist(), speed=MOVE_SPEED, is_radian=USE_RADIANS, wait=True)

        elif action_np.ndim != 1:
            print(f"[ERROR] Unexpected action tensor shape: {action_np.shape}. Expected 1D or 2D tensor.")
            continue
        else:
            print("[INFO] Action tensor only has one action, sending joint angles directly.")
            joint_angles = action_np[:6]
            print(f"[ACTION] Sending joint angles to xArm: {joint_angles}")
            # get current angles
            current_angles = arm.get_servo_angle(is_radian=USE_RADIANS)[1][:6]
            angle_diff = np.array(joint_angles) - np.array(current_angles)
            # print in 2 decimal places in degrees
            angle_diff = np.round(np.degrees(angle_diff), 2) if not USE_RADIANS else np.round(angle_diff, 2)
            # print not in the format of e+x, directly print the values
            angle_diff_str = ', '.join(f"{x:.2f}" for x in angle_diff)
            print(f"[DEBUG] Angle difference from current: {angle_diff_str} degrees")
            # if angle diff is more than 50 degrees, print a warning and continue
            if np.any(np.abs(angle_diff) > 50):
                print("[WARN] Large angle difference detected,  skipping action to prevent large movements.")
                continue
            
            

            arm.set_servo_angle(angle=joint_angles.tolist(), speed=MOVE_SPEED, is_radian=USE_RADIANS, wait=True)
            print("[INFO] Joint angles sent to xArm. Waiting 0.5s for execution...")
            time.sleep(0.5)



except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    pipeline.stop()
    print("[INFO] Cleaned up RealSense pipeline.")
