import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge

class SkeletonTrack:
    def __init__(self, keypoint_pub):
        self.keypoint_pub = keypoint_pub
        self.bridge = CvBridge()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.latest_image = None

        rospy.Subscriber("/fixed_camera/color/image_raw", Image, self.image_callback)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def get_skeleton(self, color_image):
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        keypoints = []
        if results.pose_landmarks:
            h, w, _ = color_image.shape
            self.mp_drawing.draw_landmarks(color_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            for lm in results.pose_landmarks.landmark:
                x_px = lm.x * w
                y_px = lm.y * h
                keypoints.append((x_px, y_px))

        return color_image, keypoints

    def stream(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                img, keypts = self.get_skeleton(self.latest_image.copy())

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
                        p.position.z = 0.0
                        msg.poses.append(p)

                    self.keypoint_pub.publish(msg)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            rate.sleep()

        cv2.destroyAllWindows()

def main():
    rospy.init_node('get_skeleton', anonymous=True)
    keypoint_pub = rospy.Publisher('/skeleton/keypoints_2d', PoseArray, queue_size=10)

    st = SkeletonTrack(keypoint_pub)
    st.stream()

if __name__ == '__main__':
    main()
