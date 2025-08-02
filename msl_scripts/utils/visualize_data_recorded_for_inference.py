import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


camera_name = "camera_1"

def save_camera3_frame0_images(traj_folder, output_folder):
    obs_dict_paths = [os.path.join(traj_folder, f) for f in os.listdir(traj_folder) if f.endswith(".pkl")]
    if not obs_dict_paths:
        print(f"Error: No pkl files found in folder {traj_folder}")
        return

    obs_dict_paths.sort()

    first_image = None
    for path in obs_dict_paths:
        with open(path, 'rb') as f:
            obs_dict = pickle.load(f)
        if camera_name in obs_dict:
            images = obs_dict[camera_name]
            if images.ndim == 5:
                first_image = images[0, 0]
                if first_image.shape[0] == 3:  # (C, H, W)
                    first_image = np.transpose(first_image, (1, 2, 0))
                break

    if first_image is None:
        print("No valid camera_name images found.")
        return

    H, W = first_image.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.gca()
    ax.axis("off")
    img_display = ax.imshow(first_image)

    os.makedirs(output_folder, exist_ok=True)

    plt.ion()
    plt.show(block=False)

    for idx, obs_dict_path in enumerate(obs_dict_paths):
        with open(obs_dict_path, 'rb') as f:
            obs_dict = pickle.load(f)

        if camera_name not in obs_dict:
            continue

        images = obs_dict[camera_name]
        if images.ndim != 5:
            continue

        img = images[0, 0]  # first frame

        # Convert from (C, H, W) to (H, W, C)
        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))

        # Normalize if float and convert to uint8
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, 0.0, 1.0) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        img_display.set_data(img)
        plt.draw()
        plt.pause(0.1)

        frame_path = os.path.join(output_folder, camera_name + f"_frame_{idx:04d}" + ".png")
        Image.fromarray(img).save(frame_path)

    plt.ioff()
    plt.close(fig)
    print(f"Saved {len(obs_dict_paths)} frames to: {output_folder}")


# Process all subfolders and save frames
folder_path = "/home/xarm/bags/msl_bags/inference_data/in_dis"
output_base = "./image_frames_extracted"

subfolders = [f for f in glob.glob(folder_path + "**/", recursive=True) if os.path.isdir(f)]

for subfolder in subfolders:
    name_parts = subfolder.strip("/").split(os.sep)[-2:]
    base_name = "_".join(name_parts)
    output_folder = os.path.join(output_base, base_name)

    print(f"\nExtracting frames from {subfolder}")
    save_camera3_frame0_images(subfolder, output_folder)
