import rospy
from nav_msgs.msg import Odometry
import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import xarm_sim
# import geometry_msgs/PoseStamped
from geometry_msgs.msg import PoseStamped

class XArmController:
    def __init__(self, bullet_client, xarm_sim, offset):

    # def __init__(self):
        self.bullet_client = bullet_client
        self.xarm_sim = xarm_sim
        self.offset = offset
        self.end_effector_index = 6  # xArm end-effector link index
        self.target_pose = [0.5, 0, 0.3]  # Default target pose [x, y, z]
        self.target_orientation = [1, 0, 0, 0]  # Default orientation as quaternion

        # ROS 1 Subscription to the odometry topic
        self.subscription = rospy.Subscriber(
            'vrpn_client_node/drone1/pose',
            PoseStamped,
            self.odometry_callback
        )



    

    def odometry_callback(self, msg):
        """Callback function to update target pose based on odometry."""
        self.target_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]

        # # Target pose
        # target_pose_temp = np.array([0.5773664712905884, -0.3721155524253845, 0.8803662061691284])

        # # Current end-effector pose
        # current_pose_temp = np.array([-0.0016084568566236204, -0.12631504197802784, 0.45614580154060685])

        # # Difference between target and current pose
        # difference = target_pose_temp - current_pose_temp

        # # Update target pose
        # self.target_pose = [
        #     self.target_pose[0] - difference[0],
        #     self.target_pose[1] - difference[1],
        #     self.target_pose[2] - difference[2],
        # ]


        # origin: 0.6503573060035706, -0.21394547820091248, 0.8639270663261414
        # substract target pose with the origin
        self.target_pose = [
            self.target_pose[0] - 0.6503573060035706,
            self.target_pose[1] + 0.21394547820091248,
            self.target_pose[2] - 0.8639270663261414,
        ]


        # Extract orientation as quaternion from odometry
        orientation = msg.pose.orientation
        # # set orientation as pointing down
        # orientation.x = 0
        # orientation.y = 0
        # orientation.z = 0
        # orientation.w = 1
        self.target_orientation = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        print(f"Updated target pose: {self.target_pose}")
        rospy.loginfo(f"Updated target pose: {self.target_pose}")
        # print the current robot end effector pose
        print("Current end effector pose: ", self.bullet_client.getLinkState(self.xarm_sim.xarm, self.end_effector_index)[0])


    def solve_inverse_kinematics(self):
        """Solve inverse kinematics for the target pose and orientation."""
        joint_angles = self.bullet_client.calculateInverseKinematics(
            bodyUniqueId=self.xarm_sim.xarm,
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=self.target_pose,
            targetOrientation=self.target_orientation,
        )
        return joint_angles

    def update_robot_pose(self):
        """Update robot joints based on IK solutions."""
        joint_angles = self.solve_inverse_kinematics()
        xarmNumDofs = 6
        for i in range(xarmNumDofs):
            self.bullet_client.setJointMotorControl2(
                bodyUniqueId=self.xarm_sim.xarm,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=500,
            )

def main():
    # Initialize ROS 1
    rospy.init_node('xarm_controller', anonymous=True)


    # Initialize PyBullet simulation
    p.connect(p.GUI)  # Start the PyBullet GUI
    p.setAdditionalSearchPath(pd.getDataPath())  # Set PyBullet's data path
    timeStep = 1.0 / 60.0  # Simulation timestep
    p.setTimeStep(timeStep)
    p.setGravity(0, 0, -9.8)

    # Initialize the xArm simulator
    xarm_simulator = xarm_sim.XArm6Sim(p, [0, 0, 0])  # Position the robot at [0, 0, 0]
    xarm_controller = XArmController(p, xarm_simulator, [0, 0, 0])
    # xarm_controller = XArmController()

    # Simulation loop
    rate = rospy.Rate(60)  # 60 Hz
    # try:
    while not rospy.is_shutdown():# and p.isConnected():
        xarm_controller.update_robot_pose()  # Update the robot pose
        p.stepSimulation()  # Step the PyBullet simulation
        rate.sleep()  # Maintain loop rate
    # except rospy.ROSInterruptException:
    #     rospy.loginfo('Shutting down xArm controller.')

    # Cleanup
    p.disconnect()

if __name__ == '__main__':
    main()