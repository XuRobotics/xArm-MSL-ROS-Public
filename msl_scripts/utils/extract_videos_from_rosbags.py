import rosbag
import cv2
from cv_bridge import CvBridge
import rospy
import os
import numpy as np

# Settings
bags_root_dir = '/bags/msl_bags/ALT_mutli_modality_ALT_5_5_2025/'

# the name of the folder must contain the string bag_folder_name_contain
bag_folder_name_contain = None
# bag_folder_name_contain = 'experiments_pusht' # can be empty if you want to process all bags

# Topics to extract
# topics = {
#     '/fixed_camera/color/image_raw': 'fixed_camera',
#     '/wrist_camera/color/image_raw': 'wrist_camera',
#     '/wrist_camera_2/color/image_raw': 'wrist_camera_2',
# }

topics = {
    '/fixed_camera/color/image_raw': 'fixed_camera',
    '/wrist_camera/color/image_raw': 'wrist_camera',
    # '/wrist_camera_2/color/image_raw': 'wrist_camera_2',
}

# Parameters
frame_skip_threshold = 0.0  # Percentage of pixel difference to skip frames  

# Initialize
bridge = CvBridge()

# image is 15 Hz
image_freq = 15

def is_already_processed(bag_path, output_dir):
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    for topic_prefix in topics.values():
        expected_video = f"{topic_prefix}_{bag_name}.mp4"
        expected_path = os.path.join(output_dir, expected_video)
        if os.path.exists(expected_path):
            return True
    return False

def process_bag(bag_path):
    print(f"Processing bag: {bag_path}")

    output_dir = os.path.join(os.path.dirname(bag_path), 'videos')
    os.makedirs(output_dir, exist_ok=True)

    if is_already_processed(bag_path, output_dir):
        print(f"Skipping {bag_path} because videos already exist.")
        return

    video_writers = {}
    initialized = {}
    last_frames = {}

    frame_count = {topic: 0 for topic in topics}
    skipped_count = {topic: 0 for topic in topics}

    # Extract bag name without extension
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]

    bag = rosbag.Bag(bag_path, 'r')

    try:
        total_msgs = bag.get_message_count()
        processed_msgs = 0

        for topic, msg, t in bag.read_messages(topics=topics.keys()):
            processed_msgs += 1

            if processed_msgs % 500 == 0:
                print(f"Processed {processed_msgs}/{total_msgs} messages...")

            if topic not in initialized:
                # Convert first frame to get dimensions
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                height, width = cv_img.shape[:2]

                # Create a VideoWriter for this topic
                output_filename = f"{topics[topic]}_{bag_name}.mp4"
                output_path = os.path.join(output_dir, output_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writers[topic] = cv2.VideoWriter(output_path, fourcc, image_freq, (width, height))
                initialized[topic] = True

                last_frames[topic] = cv_img
                video_writers[topic].write(cv_img)
                frame_count[topic] += 1
                continue

            # Convert ROS Image message to OpenCV image
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Compare with last frame
            if frame_skip_threshold > 0:
                diff = cv2.absdiff(last_frames[topic], cv_img)
                non_zero_count = np.count_nonzero(diff)
                total_pixels = diff.shape[0] * diff.shape[1] * diff.shape[2]

                if (non_zero_count / total_pixels) > frame_skip_threshold:
                    video_writers[topic].write(cv_img)
                    last_frames[topic] = cv_img
                    frame_count[topic] += 1
                else:
                    skipped_count[topic] += 1
            else:
                video_writers[topic].write(cv_img)
                last_frames[topic] = cv_img
                frame_count[topic] += 1
                
    finally:
        bag.close()
        for writer in video_writers.values():
            writer.release()

    print(f"Finished processing {bag_path}")
    for topic in topics:
        print(f"  {topic}: Saved {frame_count[topic]} frames, Skipped {skipped_count[topic]} frames.")

for root, dirs, files in os.walk(bags_root_dir):
    for file in files:
        # check if the folder name contains the specified string
        if bag_folder_name_contain and bag_folder_name_contain not in root:
            continue
        if file.endswith('.bag'):
            bag_path = os.path.join(root, file)
            print(f"Found bag: {bag_path}")
            process_bag(bag_path)

print("\nAll bags processed.") 
