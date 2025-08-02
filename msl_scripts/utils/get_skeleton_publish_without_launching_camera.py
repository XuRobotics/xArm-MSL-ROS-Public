import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np
import rospy
from geometry_msgs.msg import PoseArray, Pose

class SkeletonTrack:
    def __init__(self, pipeline, config, keypoint_pub):
        self.pipeline = pipeline
        self.config = config
        self.keypoint_pub = keypoint_pub

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def get_skeleton(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None, []

        color_image = np.asanyarray(color_frame.get_data())
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        keypoints = []
        if results.pose_landmarks:
            h, w, _ = color_image.shape
            self.mp_drawing.draw_landmarks(
                color_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            for lm in results.pose_landmarks.landmark:
                x_px = lm.x * w
                y_px = lm.y * h
                keypoints.append((x_px, y_px))

        return color_image, keypoints

    def stream(self):
        try:
            while not rospy.is_shutdown():
                img, keypts = self.get_skeleton()

                if img is not None:
                    cv2.imshow('Pose', img)

                if keypts:
                    msg = PoseArray()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "camera_color_frame"

                    for x, y in keypts:
                        p = Pose()
                        p.position.x = x
                        p.position.y = y
                        p.position.z = 0.0  # unused
                        # orientation left at (0, 0, 0, 0)
                        msg.poses.append(p)

                    self.keypoint_pub.publish(msg)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    rospy.init_node('get_skeleton', anonymous=True)
    
    # use PostArray since it has a header to store stamp and can be used to store multiple keypoints
    keypoint_pub = rospy.Publisher('/skeleton/keypoints_2d', PoseArray, queue_size=10)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    st = SkeletonTrack(pipeline, config, keypoint_pub)
    st.stream()

if __name__ == '__main__':
    main()
