import os
import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# output: front_img, wrist_img, state_vec, where state_vec = [x, y, z, rot_6d, gripper]
# where x, y, z are the position in mm
# where rot_6d is the first 6 elements of the rotation matrix in 6D representation
# where gripper is the opening value in the original range of 0-850!


# === CONFIG ===
ROOT_BAG_DIR = "/bags/msl_bags/OWR-bags-7-28-2025-robot-demo-for-inference"
SAVE_ROOT = os.path.join(ROOT_BAG_DIR, "extracted_images_and_poses_OWR")
FRONT_TOPIC = "/fixed_camera/color/image_raw"
WRIST_TOPIC = "/wrist_camera/color/image_raw"
POSE_TOPIC = "/robot_end_effector_pose"
GRIPPER_TOPIC = "/robot_end_effector_opening"
SYNC_THRESH = 0.05


def quat_to_six_d(quat):
    rot_matrix = R.from_quat(quat).as_matrix()
    return rot_matrix[:, :2].flatten()


def pose_to_pos_6d(pose_msg):
    pos = pose_msg.pose.position
    ori = pose_msg.pose.orientation
    quat = [ori.x, ori.y, ori.z, ori.w]
    rot_6d = quat_to_six_d(quat)
    return [pos.x, pos.y, pos.z] + rot_6d.tolist()  # IMPORTANT: (x, y, z) then 6D


def extract_first_synced_sample(bag_path):
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()

    front_images = []
    wrist_images = []
    poses = []
    grippers = []

    for topic, msg, t in bag.read_messages(topics=[FRONT_TOPIC, WRIST_TOPIC, POSE_TOPIC, GRIPPER_TOPIC]):
        ts = t.to_sec()
        if topic == FRONT_TOPIC:
            front_images.append((ts, bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')))
        elif topic == WRIST_TOPIC:
            wrist_images.append((ts, bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')))
        elif topic == POSE_TOPIC:
            poses.append((ts, pose_to_pos_6d(msg)))
        elif topic == GRIPPER_TOPIC:
            grippers.append((ts, msg.data))

    bag.close()

    for ft, fimg in front_images:
        for wt, wimg in wrist_images:
            if abs(ft - wt) < SYNC_THRESH:
                for pt, pose in poses:
                    if abs(pt - ft) < SYNC_THRESH:
                        for gt, gval in grippers:
                            if abs(gt - ft) < SYNC_THRESH:
                                return fimg, wimg, pose + [gval]
    return None


if __name__ == "__main__":
    os.makedirs(SAVE_ROOT, exist_ok=True)

    bag_files = [os.path.join(dp, f) for dp, _, fn in os.walk(ROOT_BAG_DIR)
                 for f in fn if f.endswith(".bag")]

    print(f"Found {len(bag_files)} bag files.")

    for bag_path in tqdm(bag_files, desc="Extracting first frames"):
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        save_dir = os.path.join(SAVE_ROOT, bag_name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            result = extract_first_synced_sample(bag_path)
        except Exception as e:
            print(f"[{bag_name}] Failed: {e}")
            continue

        if result is None:
            print(f"[{bag_name}] No synced sample found")
            continue

        front_img, wrist_img, state_vec = result
        front_img = cv2.resize(front_img, (1280, 720))
        wrist_img = cv2.resize(wrist_img, (1280, 720))

        cv2.imwrite(os.path.join(save_dir, "camera_0_frame_0000.png"), cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "camera_1_frame_0000.png"), cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(save_dir, "state.npy"), np.array(state_vec, dtype=np.float32))  # shape (10,)
        print(f"[{bag_name}] Extracted and saved.")
        print(f"saved state vector: {state_vec}")
