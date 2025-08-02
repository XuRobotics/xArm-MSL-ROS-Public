import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import os
from transformers import AutoTokenizer
import inspect
import traceback


# ----------------------------------------
print("[INFO] Loading Pi0 policy...")
policy = PI0Policy.from_pretrained("lerobot/pi0")
print("[INFO] Policy loaded successfully.")
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
print("[INFO] Tokenizer loaded successfully.")

# ----------------------------------------
goal = ["pick up the cup"]
print(f"[INFO] Goal set to: {goal[0]}")

# ----------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])
print("[INFO] Preprocessing pipeline initialized.")

# ----------------------------------------
print("[INFO] Starting RealSense pipeline...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ----------------------------------------
# Direct Pi0 model call
def select_action_direct(policy, image_tensor, goal_text):
    model = policy.model
    device = next(model.parameters()).device

    # Tokenize goal
    tokens = tokenizer(goal_text, return_tensors="pt", padding=True, truncation=True)

    input_ids = tokens["input_ids"].to(device)
    attn_mask = tokens["attention_mask"].to(device)
    # Ensure attention mask and input IDs match in shape
    seq_len = min(input_ids.shape[1], attn_mask.shape[1])
    input_ids = input_ids[:, :seq_len]
    attn_mask = attn_mask[:, :seq_len]
    
    # Ensure 4D vision tensor
    vision = image_tensor.to(dtype=torch.float32, device=device)
    if vision.ndim == 3:  # shape [C, H, W]
        vision = vision.unsqueeze(0)  # now [1, C, H, W]

    # Batch size correctly derived from vision
    batch_size = vision.shape[0]

    # Create image masks and dummy inputs
    img_masks = torch.ones((batch_size, 1, 1, 1), dtype=torch.bool, device=device)
    state = torch.zeros(batch_size, policy.config.max_state_dim, device=device)
    actions = torch.zeros(batch_size, policy.config.max_action_dim, device=device)

    # Run inference
    with torch.no_grad():
        assert vision.ndim == 4, f"Expected 4D input, got {vision.shape}"

        action_pred = model.forward(
            vision,
            img_masks,
            input_ids,
            attn_mask,
            state,
            actions
        )
    return action_pred


try:
    pipeline.start(config)
    print("[INFO] RealSense streaming started.")
    os.makedirs("debug_output", exist_ok=True)

    frame_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("[WARN] No frame received. Skipping.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        frame_count += 1
        print(f"[DEBUG] Frame {frame_count} received.")

        # Preprocess
        image_pil = Image.fromarray(color_image).convert("RGB")
        image_tensor = preprocess(image_pil)
        image_tensor = image_tensor.to(dtype=torch.float32, device=next(policy.parameters()).device)
        print(f"[DEBUG] Image preprocessed. Shape: {image_tensor.shape}")

        # Save preprocessed image for debug
        input_vis = image_tensor.detach().cpu() * 0.5 + 0.5  # [C, H, W]
        input_vis = input_vis.permute(1, 2, 0).numpy()        # [H, W, C]
        input_vis = (input_vis * 255).astype(np.uint8)
        cv2.imwrite(f"/bags/input_to_pi0_{frame_count:04}.jpg", input_vis)

        # Inference (bypassing LeRobot registry)
        print(f"[DEBUG] Running direct model inference...")
        try:
            action = select_action_direct(policy, image_tensor, goal[0])
            print(f"[RESULT] Predicted action: {action}")
        except Exception as e:
            print("[ERROR] Policy inference failed:")
            traceback.print_exc()
            break

        # Overlay action on original image
        vis_image = color_image.copy()
        y_offset = 30
        try:
            action_np = action.detach().cpu().numpy().round(3).tolist()
            text = f"action: {action_np}"
        except Exception as e:
            text = f"action: [error: {e}]"
        cv2.putText(vis_image, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save output
        cv2.imwrite(f"/bags/inference_output_{frame_count:04}.jpg", vis_image)
        print(f"[INFO] Saved frame {frame_count} with action overlay.")

        if frame_count >= 10:
            print("[INFO] Reached 10 frames. Exiting.")
            break

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    print("[INFO] Cleaning up...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Shutdown complete.")
