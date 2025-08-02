"""
MIT License

Copyright (c) 2025 Xu Liu, Chengyang He, Gadiel Sznaier Camps at 
Multi-Robot Systems Lab, Stanford University (with Github Copilot)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import example.wrapper.xarm6.xarm_sim as xarm_sim

class XArmController(Node):
    def __init__(self, bullet_client, xarm_sim, offset):
        super().__init__('xarm_controller')
        self.bullet_client = bullet_client
        self.xarm_sim = xarm_sim
        self.offset = offset
        self.end_effector_index = 6  # xArm end-effector link index
        self.target_pose = [0.5, 0, 0.3]  # Default target pose [x, y, z]
        self.target_orientation = [1, 0, 0, 0]  # Default orientation as quaternion

        # ROS 2 Subscription to the odometry topic
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10  # QoS Profile with depth of 10
        )

    def odometry_callback(self, msg):
        """Callback function to update target pose based on odometry."""
        self.target_pose = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        # Extract orientation as quaternion from odometry
        orientation = msg.pose.pose.orientation
        self.target_orientation = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        self.get_logger().info(f"Updated target pose: {self.target_pose}")
        # get the current gripper pose and print it 
        print("getting gripper pose")
        gripper_pose = self.bullet_client.getLinkState(self.xarm_sim.xarm, 6)[0]
        self.get_logger().info(f"Current gripper pose: {gripper_pose}")



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


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Initialize PyBullet simulation
    p.connect(p.GUI)  # Start the PyBullet GUI
    p.setAdditionalSearchPath(pd.getDataPath())  # Set PyBullet's data path
    timeStep = 1.0 / 60.0  # Simulation timestep
    p.setTimeStep(timeStep)
    p.setGravity(0, 0, -9.8)

    # Initialize the xArm simulator
    xarm_simulator = xarm_sim.XArm6Sim(p, [0, 0, 0])  # Position the robot at [0, 0, 0]
    xarm_controller = XArmController(p, xarm_simulator, [0, 0, 0])

    # Simulation loop
    # try:
    while rclpy.ok() and p.isConnected():
        xarm_controller.update_robot_pose()  # Update the robot pose
        p.stepSimulation()  # Step the PyBullet simulation
        time.sleep(1.0 / 60.0)  # Maintain real-time simulation
        rclpy.spin_once(xarm_controller, timeout_sec=0)  # Process ROS callbacks
    # except KeyboardInterrupt:
    #     xarm_controller.get_logger().info('Shutting down xArm controller.')

    # Cleanup
    xarm_controller.destroy_node()
    rclpy.shutdown()
    p.disconnect()


if __name__ == '__main__':
    main()
