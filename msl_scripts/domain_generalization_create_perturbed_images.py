from PIL import Image, ImageEnhance
import numpy as np
import os
import shutil
import cv2


ROOT_DIR = "/home/xarm/bags/msl_bags/IMPORTANT-distribution-pick-and-place-raw-bags-30"



# ======================================= 
#### REPLACE CUP COLOR
# Detects cups using YOLOv8 segmentation (very low confidence threshold to include faint/partial cups).
# Replaces the color of all pixels in cup masks with teal ([B, G, R] = [180, 200, 0]).
# Leaves background and non-cup objects untouched.



#### GAUSSIAN NOISE ADDITION
# np.random.normal(mean=0.0, std=sigma, size=image.shape). 
# Random values drawn from a Gaussian distribution with 0 mean and a chosen standard deviation (sigma). Each pixel’s R, G, B channels get an independent random offset. 
# The noise array is added element-wise to the image’s pixel array. 
# Clipped to the valid intensity range [0, 255] to avoid wrap-around or overflow.
# noise_sigma: Standard deviation of the Gaussian noise.
# --300: SNR ≈ -7.4 dB (noise is stronger than the signal — heavily corrupted image)
# --200: SNR ≈ -3.9 dB → heavy noise, but image features still partially visible
# --100: SNR ≈ +2.1 dB → strong noise, but major image structure remains usable



#### SIMULATE ROTATED CAMERA VIEW
# Uses  a homography (perspective transform) derived from the given yaw, pitch, and roll angles:
# Using the intrinsics and rotation, computes a 3×3 homography H = K · R · K⁻¹
# The homography represents how a plane (the image) appears under the camera rotation (pure rotation about the optical center). 
# The image is warped with OpenCV’s warpPerspective using H
# A rotation will leave black triangles at the image borders (where no original pixels map). Fills those areas smoothly:
# --Finds the median color of the original image’s border 
# --Uses that as a background fill color.
# --Creates a mask of valid (non-black) pixels after warping, then blurs this mask
# --Alpha-blends the warped image with a solid image of the median color
# --This produces a smooth transition at the edges instead of sharp borders.


### REPLACE BACKGROUND COLOR
# Gray detection: Convert to HSV; gray = low saturation (S: 0–50), value range 30–220
# Yellow detection: Convert to Lab; yellow = high L, neutral a, high b (e.g., b in 90–210)
# Create masks with cv2.inRange, replace masked pixels with background fill
# Result: Uniform background, gray tables and yellow paper removed



#### IN-PLANE ROTATION
# Rotates image around center using PIL’s Image.rotate with expand=False
# Keeps original canvas size — corners may get cut off; black background fills appear
# Simulates in-plane camera tilt (like rotating the camera around the optical axis)
# Output resolution is unchanged; black edges may appear from clipped corners

#### TRANSLATION (SHIFTING)
# Shifts image by (translate_x, translate_y) pixels
# Crops appropriate region from the original image and pastes it onto a new black canvas
# Positive translate_x: shift right; positive translate_y: shift down
# Leaves black bands where image data was shifted out of view
# Simulates slight camera movement or misalignment

#### SCALING (ZOOM IN / OUT)
# Applies uniform scaling using bicubic interpolation
# -- If scale_factor > 1.0: zoom in → enlarge and center-crop to original size
# -- If scale_factor < 1.0: zoom out → shrink and center on black canvas
# Simulates camera moving closer or farther
# Maintains original resolution; results in cropping or padding depending on zoom

#### BRIGHTNESS CHANGE
# Adjusts brightness using ImageEnhance.Brightness
# Multiplies all pixel values by intensity_factor
# -- 1.0 → original brightness
# -- >1.0 → brighter (e.g., 1.2 = +20%)
# -- <1.0 → darker (e.g., 0.8 = -20%)
# -- 0.0 → completely black
# Simulates lighting variation in a linear and global manner




def replace_cup_color(img_np, 
                                model=None, 
                                teal_bgr=(180, 200, 0)):
    """
    Segments cups using YOLOv8 segmentation and replaces their color with teal.

    Args:
        img_np (np.ndarray): Input image (numpy array, BGR).
        model (YOLO): Preloaded YOLO segmentation model (optional).
        teal_bgr (tuple): Teal color (B, G, R).

    Returns:
        np.ndarray: Image with cup regions colored teal.
    """
    # Ensure model is loaded
    if model is None:
        model = YOLO("yolov8x-seg.pt")
    
    # Convert to RGB for YOLO (if needed)
    img_for_yolo = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    results = model(img_for_yolo, conf=0.1)[0]

    # Copy image for output
    img_out = img_np.copy()
    for i, cls in enumerate(results.boxes.cls):
        class_id = int(cls.item())
        class_name = model.names[class_id]
        if class_name == "cup":
            mask = results.masks.data[i].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            mask = cv2.resize(mask, (img_out.shape[1], img_out.shape[0]))
            img_out[mask == 1] = teal_bgr  # BGR order

    return img_out

def process_cups_in_directory(input_dir, output_dir, 
                              teal_bgr=(180, 200, 0),
                              extensions={'.png', '.jpg', '.jpeg', '.bmp'}):
    """
    Recursively colors cups teal in all images in input_dir, saves to output_dir.

    Args:
        input_dir (str): Relative path to input directory.
        output_dir (str): Relative path to output directory.
        teal_bgr (tuple): Teal color in BGR.
        extensions (set): File extensions to process.
        ROOT_DIR (str): Root directory for input/output.
    """

    input_dir = os.path.join(ROOT_DIR, input_dir)
    output_dir = os.path.join(ROOT_DIR, output_dir)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load model once for efficiency
    model = YOLO("yolov8x-seg.pt")

    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_path) if rel_path != '.' else output_dir
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_root, file)
            if ext in extensions:
                print(f"Coloring cups in: {src_path}")
                img = Image.open(src_path).convert("RGB")
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                teal_img_bgr = replace_cup_color(
                    img_bgr, model=model, teal_bgr=teal_bgr
                )
                teal_img_rgb = cv2.cvtColor(teal_img_bgr, cv2.COLOR_BGR2RGB)
                out_img = Image.fromarray(teal_img_rgb)
                out_img.save(dst_path)
            else:
                shutil.copy2(src_path, dst_path)


def replace_background_and_yellow(img_np,
                                   bg_color_rgb=(180, 220, 255),
                                   replace_gray=True,
                                   replace_yellow=True):
    bg_fill = np.full_like(img_np, bg_color_rgb)

    if replace_gray:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        lower_gray = np.array([0, 0, 30])
        upper_gray = np.array([180, 50, 220])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        img_np = np.where(gray_mask[:, :, None] == 255, bg_fill, img_np)

    if replace_yellow:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # test 3: most aggressive yellow
        # lower_yellow = np.array([120, 100, 90])
        # upper_yellow = np.array([255, 160, 210])
        # lower_yellow = np.array([140, 100, 90])
        # upper_yellow = np.array([255, 160, 210])
        # # test 2: more aggressive yellow
        # lower_yellow = np.array([150, 100, 100])
        # upper_yellow = np.array([255, 160, 190]) 
        # # test 1: least aggressive yellow
        lower_yellow = np.array([170, 110, 90])
        upper_yellow = np.array([255, 170, 170])
        yellow_mask = cv2.inRange(lab, lower_yellow, upper_yellow)
        img_np = np.where(yellow_mask[:, :, None] == 255, bg_fill, img_np)

    return img_np

def clean_backgrounds(input_dir, output_dir,
                      bg_color_rgb=(180, 220, 255),
                      replace_gray=True,
                      replace_yellow=True,
                      extensions={'.png', '.jpg', '.jpeg', '.bmp'}):
    """
    Recursively cleans gray/yellow backgrounds from images in input_dir
    and writes the results to output_dir.
    """
    # join ROOT_DIR with input_dir and output_dir
    input_dir = os.path.join(ROOT_DIR, input_dir)
    output_dir = os.path.join(ROOT_DIR, output_dir)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_path) if rel_path != '.' else output_dir
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_root, file)

            if ext in extensions:
                print(f"Cleaning: {src_path}")
                img = Image.open(src_path).convert("RGB")
                img_np = np.array(img)

                cleaned_np = replace_background_and_yellow(
                    img_np,
                    bg_color_rgb=bg_color_rgb,
                    replace_gray=replace_gray,
                    replace_yellow=replace_yellow
                )
                cleaned_img = Image.fromarray(cleaned_np)
                cleaned_img.save(dst_path)
            else:
                shutil.copy2(src_path, dst_path)


def simulate_view_rotation(img_pil, yaw_deg=15, pitch_deg=0, roll_deg=0, fov_deg=60, blend_width=20):
    """
    Applies a 3D perspective warp and uses feathered blending to smoothly fill border regions.
    The background color is taken from the median of the border pixels.
    """
    img = np.array(img_pil)
    h, w = img.shape[:2]

    # === Median color from image border ===
    border_pixels = np.vstack([
        img[0, :, :],
        img[-1, :, :],
        img[:, 0, :],
        img[:, -1, :]
    ])
    median_color = np.median(border_pixels, axis=0).astype(np.uint8)
    median_color_bgr = tuple(int(c) for c in median_color[::-1])

    # === Intrinsics and rotation ===
    yaw, pitch, roll = map(np.radians, [yaw_deg, pitch_deg, roll_deg])
    fov = np.radians(fov_deg)
    focal = 0.5 * w / np.tan(fov / 2)
    K = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]
    ])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    H = K @ R @ np.linalg.inv(K)

    # === Warp image and create validity mask ===
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Generate mask: valid (non-black) pixels after warp
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    mask = (gray > 1).astype(np.uint8) * 255  # binary mask
    mask_blurred = cv2.GaussianBlur(mask, (2 * blend_width + 1, 2 * blend_width + 1), 0)
    alpha = mask_blurred.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)  # shape (H, W, 1)

    # Background image: filled with median color
    bg = np.full_like(warped, median_color, dtype=np.uint8)

    # === Blend foreground (warped image) and background ===
    blended = warped.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def perturb_images(input_dir, output_dir,
                   add_gaussian_noise=False, noise_sigma=15.0,
                   change_intensity=False, intensity_factor=1.2,
                   rotate=False, rotation_angle=15.0,
                   translate=False, translate_x=10, translate_y=10,
                   scale=False, scale_factor=1.2,
                   simulate_view=False, view_yaw=15.0, view_pitch=0.0, view_roll=0.0,
                   perturb_only_camera0=True):
    """
    Applies specified perturbations to all images in the directory structure of `input_dir` 
    and saves them to `output_dir`, preserving the folder structure. Non-image files are copied as-is.
    
    Parameters:
    - input_dir (str): Path to the source directory containing subfolders with images (and possibly state files).
    - output_dir (str): Path to the target directory where perturbed images will be saved.
    - add_gaussian_noise (bool): If True, add Gaussian noise to each image.
    - noise_sigma (float): Standard deviation of Gaussian noise (0-255 scale for pixel intensity).
    - change_intensity (bool): If True, adjust image brightness by a factor.
    - intensity_factor (float): Brightness factor (1.0 = no change, >1 = brighter, <1 = darker).
    - rotate (bool): If True, rotate each image by `rotation_angle` degrees.
    - rotation_angle (float): Degrees to rotate the image (positive values = counter-clockwise rotation).
    - translate (bool): If True, translate (shift) each image by (`translate_x`, `translate_y`) pixels.
    - translate_x (int): Horizontal shift in pixels (positive = shift right, negative = shift left).
    - translate_y (int): Vertical shift in pixels (positive = shift down, negative = shift up).
    - scale (bool): If True, scale each image by `scale_factor`.
    - scale_factor (float): Scaling factor for resizing (e.g., 1.2 = 120% of original size, 0.8 = 80% of original size).
    - perturb_only_camera0 (bool): If True, only images with 'camera_0_' in their name will be perturbed;
                                  all others are copied unchanged.
    """
    # Delete the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing output directory: {output_dir}")   
    os.makedirs(output_dir, exist_ok=True)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("PERTURBATION SETTINGS:")
    if simulate_view:
        print(f"Simulating 3D view with yaw={view_yaw}, pitch={view_pitch}, roll={view_roll}")
    else:
        print("Not simulating rotated view.")
    if add_gaussian_noise:
        print(f"Adding Gaussian noise with sigma: {noise_sigma}")
    else:
        print("Not adding Gaussian noise.")
    if change_intensity:
        print(f"Changing intensity with factor: {intensity_factor}")
    else:
        print("Not changing intensity.")
    if rotate:
        print(f"Rotating images by: {rotation_angle} degrees")
    else:
        print("Not rotating images.")
    if translate:
        print(f"Translating images by: ({translate_x}, {translate_y}) pixels")
    else:
        print("Not translating images.")
    if scale:
        print(f"Scaling images by factor: {scale_factor}")
    else:
        print("Not scaling images.")
    # print perturbation params
    if perturb_only_camera0:
        print("Only perturbing images with 'camera_0_' in their filename.")
    else:
        print("Perturbing all images regardless of camera name.")

    # add the ROOT_DIR to the input_dir if not already set
    input_dir = os.path.join(ROOT_DIR, input_dir) 
    output_dir = os.path.join(ROOT_DIR, output_dir)
    print(f"Processing images from: {input_dir}")
    # Walk through all files and subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        print(f"Processing directory: {root}")
        # Determine the corresponding path in the output directory
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == '.':
            rel_path = ''
        target_root = os.path.join(output_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)  # create subfolder in output
        
        for filename in files:
            print(f"Processing file: {filename}")
            src_path = os.path.join(root, filename)
            dst_path = os.path.join(target_root, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in ['.png', '.jpg', '.jpeg', '.bmp']:  # image file extensions

                # ========== Only perturb camera_0_ images if flag is set ==========
                if perturb_only_camera0 and 'camera_0_' not in filename:
                    shutil.copy2(src_path, dst_path)
                    continue

                # Open image and convert to RGB (to ensure 3-channel format)
                img = Image.open(src_path).convert("RGB")
                # 0. Simulate rotated camera view
                if simulate_view:
                    img = simulate_view_rotation(
                        img,
                        yaw_deg=view_yaw,
                        pitch_deg=view_pitch,
                        roll_deg=view_roll
                    )

                    # Crop center and rescale to remove edge warping
                    final_center_crop_ratio = 0.9  # change if needed
                    width, height = img.size
                    crop_w = int(width * final_center_crop_ratio)
                    crop_h = int(height * final_center_crop_ratio)
                    left = (width - crop_w) // 2
                    top = (height - crop_h) // 2
                    right = left + crop_w
                    bottom = top + crop_h

                    img = img.crop((left, top, right, bottom))
                    img = img.resize((width, height), resample=Image.Resampling.BICUBIC)

                # 1. Scaling (zoom in/out)
                if scale and scale_factor != 1.0:
                    width, height = img.size
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    # Resize the image with high-quality resampling
                    img_resized = img.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
                    if scale_factor > 1.0:
                        # If zooming in (image is larger), crop the center to original size
                        left = (new_width - width) // 2
                        top = (new_height - height) // 2
                        right = left + width
                        bottom = top + height
                        img = img_resized.crop((left, top, right, bottom))
                    else:
                        # If zooming out (image is smaller), place it on a black canvas of original size
                        canvas = Image.new("RGB", (width, height))  # default fill is black
                        left = (width - new_width) // 2
                        top = (height - new_height) // 2
                        canvas.paste(img_resized, (left, top))
                        img = canvas
                
                # 2. Rotation
                if rotate and rotation_angle != 0:
                    # Rotate around center without expanding canvas (fills corners with black)
                    img = img.rotate(rotation_angle, resample=Image.Resampling.BICUBIC, expand=False)
                
                # 3. Translation (shifting)
                if translate and (translate_x != 0 or translate_y != 0):
                    width, height = img.size
                    # Create a new black image for the canvas
                    translated_img = Image.new("RGB", (width, height))
                    # Determine source region to crop from the original image (to handle negative shifts)
                    src_x = max(0, -translate_x)        # if shifting left (negative x), skip that many pixels from left
                    src_y = max(0, -translate_y)        # if shifting up (negative y), skip that many pixels from top
                    src_w = width - max(0, translate_x)  # if shifting right, skip that many pixels from right
                    src_h = height - max(0, translate_y) # if shifting down, skip that many pixels from bottom
                    # Crop the part of the image that will remain within the view after translation
                    cropped_region = img.crop((src_x, src_y, src_w, src_h))
                    # Determine destination placement on the canvas
                    dst_x = max(0, translate_x)  # if shifting right, start past the left border
                    dst_y = max(0, translate_y)  # if shifting down, start past the top border
                    # Paste the cropped region onto the black canvas at the new location
                    translated_img.paste(cropped_region, (dst_x, dst_y))
                    img = translated_img
                
                # 4. Brightness/Intensity adjustment
                if change_intensity and intensity_factor != 1.0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(intensity_factor)
                
                # 5. Gaussian noise addition
                if add_gaussian_noise and noise_sigma > 0:
                    # Convert image to NumPy array for pixel-level manipulation
                    img_array = np.array(img).astype(np.float32)
                    # Add Gaussian noise
                    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=img_array.shape)
                    img_array += noise
                    # Clip values to valid range [0, 255] and convert back to unsigned 8-bit
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    # Convert back to PIL Image
                    img = Image.fromarray(img_array)
                
                # Save the perturbed image to the output path
                img.save(dst_path)
            
            else:
                # For non-image files (e.g., .npy state files), simply copy them to the new location
                shutil.copy2(src_path, dst_path)


# # ========== USAGE EXAMPLES ==========

# # 1. All Perturbations (Defaults)
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=True,
#     change_intensity=True,
#     rotate=True,
#     translate=True,
#     scale=True,
#     perturb_only_camera0=False  # Apply to all images, not just camera_0_
# )

# # 1. All Perturbations BUT ROTATION (Defaults)
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=True,
#     change_intensity=True,
#     rotate=True,
#     translate=True,
#     scale=True,
#     perturb_only_camera0=False  # Apply to all images, not just camera_0_
# )

# # 1. All Perturbations with Pitch 10.0 degrees 
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=True,
#     change_intensity=True,
#     rotate=False,
#     translate=True,
#     scale=True,
#     simulate_view=True,
#     view_yaw=0.0,
#     view_pitch=10.0,
#     view_roll=0.0,
#     perturb_only_camera0=True
# )

# # 1. All Perturbations with ROLL 10.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=True,
#     change_intensity=True,
#     rotate=False,
#     translate=True,
#     scale=True,
#     simulate_view=True,
#     view_yaw=0.0,
#     view_pitch=0.0,
#     view_roll=10.0,
#     perturb_only_camera0=True
# )


# # 1. ROLL 10.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=0.0,
#     view_pitch=0.0,
#     view_roll=10.0,
#     perturb_only_camera0=True
# )

# # 1. PITCH 10.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=0.0,
#     view_pitch=10.0,
#     view_roll=0.0,
#     perturb_only_camera0=True
# )

# # 1. Yaw 5.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=5.0,
#     view_pitch=0.0,
#     view_roll=0.0,
#     perturb_only_camera0=True
# )

# # 1. Yaw 10.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=10.0,
#     view_pitch=0.0,
#     view_roll=0.0,
#     perturb_only_camera0=True
# )


# # 1. roll 5.0 degrees, pitch 5.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=0.0,
#     view_pitch=5.0,
#     view_roll=5.0,
#     perturb_only_camera0=True
# )


# # 1. Yaw 15.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=15.0,
#     view_pitch=0.0,
#     view_roll=0.0,
#     perturb_only_camera0=True
# )


# # 1. Yaw 13.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=13.0,
#     view_pitch=0.0,
#     view_roll=0.0,
#     perturb_only_camera0=True
# )


# # 1. Yaw -10.0 degrees
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=False,
#     change_intensity=False,
#     rotate=False,
#     translate=False,
#     scale=False,
#     simulate_view=True,
#     view_yaw=-10.0,
#     view_pitch=0.0,
#     view_roll=0.0,
#     perturb_only_camera0=True
# )




# # Background color replacement
# clean_backgrounds(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed",
#     bg_color_rgb=(180, 220, 255),        # light blue background replacement
#     replace_gray=True,                   # remove gray table
#     replace_yellow=True                 # remove yellow paper
# )





# # Background color replacement + Gaussian noise
# clean_backgrounds(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_TEMP",
#     bg_color_rgb=(180, 220, 255),        # light blue background replacement
#     replace_gray=True,                   # remove gray table
#     replace_yellow=True                 # remove yellow paper
# )
# print("Now applying Gaussian noise...")
# # sleep for 2 seconds 
# import time
# time.sleep(2)
# perturb_images(
#     input_dir="extracted_images_and_poses_perturbed_TEMP",
#     output_dir="extracted_images_and_poses_perturbed",
#     add_gaussian_noise=True,
#     perturb_only_camera0=False,
#     noise_sigma=300.0  # TOO_BIG_GAUSSIAN: 400 # BIGGEST_GAUSSAN: 300 # BIGGER_GAUSSIAN: 200.0 # BIG_GAUSSIAN: 100.0 # NORMAL: 50.0
# )



# # Replace Color of Cup
# from ultralytics import YOLO
# process_cups_in_directory(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed")



# Replace Color of Cup + Background Color Replacement + Gaussian noise
# Background color replacement + Gaussian noise
clean_backgrounds(
    input_dir="extracted_images_and_poses_ORIGINAL",
    output_dir="extracted_images_and_poses_perturbed_TEMP",
    bg_color_rgb=(180, 220, 255),        # light blue background replacement
    replace_gray=True,                   # remove gray table
    replace_yellow=True                 # remove yellow paper
)
print("Now replacing cup color...")
from ultralytics import YOLO
process_cups_in_directory(
    input_dir="extracted_images_and_poses_perturbed_TEMP",
    output_dir="extracted_images_and_poses_perturbed_TEMP2")
print("Now applying Gaussian noise...")
# sleep for 2 seconds 
import time
time.sleep(2)
perturb_images(
    input_dir="extracted_images_and_poses_perturbed_TEMP2",
    output_dir="extracted_images_and_poses_perturbed",
    add_gaussian_noise=True,
    perturb_only_camera0=False,
    noise_sigma=200.0  # TOO_BIG_GAUSSIAN: 400 # BIGGEST_GAUSSAN: 300 # BIGGER_GAUSSIAN: 200.0 # BIG_GAUSSIAN: 100.0 # NORMAL: 50.0
)























































############################# NOT USED YET
# # 2. Only Gaussian Noise
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_noise",
#     add_gaussian_noise=True,
#     noise_sigma=25.0
# )

# # 3. Only Brightness Change
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_bright",
#     change_intensity=True,
#     intensity_factor=1.5
# )

# # 4. Only Rotation
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_rotate",
#     rotate=True,
#     rotation_angle=30.0
# )

# # 5. Only Translation
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_translate",
#     translate=True,
#     translate_x=20,
#     translate_y=-10
# )

# # 6. Only Scaling
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_scale",
#     scale=True,
#     scale_factor=0.8
# )

# # 7. Combine: Strong Noise + Large Rotation
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_noise_rotate",
#     add_gaussian_noise=True, noise_sigma=40.0,
#     rotate=True, rotation_angle=90.0
# )

# # 8. Combine: Scale Down and Brighten
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_scale_bright",
#     scale=True, scale_factor=0.7,
#     change_intensity=True, intensity_factor=1.4
# )

# # 9. Combine: Translate + Noise + Darken
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_translate_noise_dark",
#     translate=True, translate_x=12, translate_y=8,
#     add_gaussian_noise=True, noise_sigma=20,
#     change_intensity=True, intensity_factor=0.7
# )

# # 10. No Perturbations (Just Copy Structure, e.g. for Testing)
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_copy"
# )

# # 11. Simulate a rotated camera view
# perturb_images(
#     input_dir="extracted_images_and_poses_ORIGINAL",
#     output_dir="extracted_images_and_poses_perturbed_viewrotated",
#     simulate_view=True,
#     view_yaw=20.0,
#     view_pitch=10.0,
#     view_roll=5.0,
#     perturb_only_camera0=False
# )