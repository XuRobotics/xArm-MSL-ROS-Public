import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

def visualize_images(root_dir, root_dir2=None, save=True):
    def get_image_pairs(root):
        subdirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
        pairs = []
        for subdir in subdirs:
            cam0 = os.path.join(subdir, "camera_0_frame_0000.png")
            cam1 = os.path.join(subdir, "camera_1_frame_0000.png")
            if os.path.exists(cam0) and os.path.exists(cam1):
                img0 = Image.open(cam0).copy()
                img1 = Image.open(cam1).copy()
                combined = Image.new("RGB", (img0.width + img1.width, max(img0.height, img1.height)))
                combined.paste(img0, (0, 0))
                combined.paste(img1, (img0.width, 0))
                pairs.append((combined, os.path.basename(subdir)))
            else:
                print(f"Missing images in: {subdir}")
        return pairs

    # Prepare output directory from root_dir2's 2nd last segment
    if root_dir2:
        parts = os.path.normpath(root_dir2).split(os.sep)
        output_folder = parts[-2] if len(parts) >= 2 else "visualizations"
    else:
        output_folder = "visualizations"

    os.makedirs(output_folder, exist_ok=True)
    
    # Get image pairs
    pairs1 = get_image_pairs(root_dir)
    pairs2 = get_image_pairs(root_dir2) if root_dir2 else []

    max_len = max(len(pairs1), len(pairs2))
    pairs1.extend([(None, "")] * (max_len - len(pairs1)))
    pairs2.extend([(None, "")] * (max_len - len(pairs2)))

    # Setup 2-row, 1-column layout
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.3)

    im_handles = [None, None]

    # Hide all axes borders once
    for ax in axs:
        ax.axis("off")

    # Iterate through image pairs
    for i in range(max_len):
        for row, (pairs, label_prefix) in enumerate(zip([pairs1, pairs2], ["Set1", "Set2"])):
            img, label = pairs[i]
            ax = axs[row]
            if img:
                if im_handles[row] is None:
                    im_handles[row] = ax.imshow(img)
                else:
                    im_handles[row].set_data(img)
                ax.set_title(f"{label_prefix}: {label}", fontsize=12)
            else:
                ax.set_title("")

        fig.canvas.draw()
        plt.pause(0.001)  # let GUI update
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(output_folder, f"visualization_{timestamp}.png")
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

        plt.pause(0.5)

    plt.close(fig)


if __name__ == "__main__":
    root_dir = "/home/sam/Desktop/domain_gen_july_2025/july_27_2025/extracted_images_and_poses_ORIGINAL"
    root_dir2 = "/home/sam/Desktop/domain_gen_july_2025/july_27_2025/Replace-Cup-Color-Perturb-BIGGER-Gaussian-BEST/extracted_images_and_poses_perturbed"
    # root_dir2 = "/home/sam/Desktop/domain_gen_july_2025/july_27_2025/Color-Perturb-MOST-Aggressive-EVEN-MORE-Add-BIGGEST-Guassian-BEST/extracted_images_and_poses_perturbed"
    visualize_images(root_dir, root_dir2)
