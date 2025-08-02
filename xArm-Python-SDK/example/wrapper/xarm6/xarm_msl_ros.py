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

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32  # Import Float32 message type
import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
# import geometry_msgs/PoseStamped
from geometry_msgs.msg import PoseStamped, Pose
from pathlib import Path
from sak.URDFutils import URDFutils
from pydrake.math import RotationMatrix

from pydrake.all import *
import pydrake.multibody.plant as pmp
# scipy import R
from scipy.spatial.transform import Rotation as R
import sys
import os
# append xarm stuff, which is ../../../xarm/
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import xarm_sim
from xarm.wrapper import XArmAPI


# FULL API DOWNLOAD LINK: http://download.ufactory.cc/xarm/en/xArm%20Developer%20Manual.pdf?v=1600992000052

class XArmController:
    def __init__(self, run_sim = True):
        #################### IMPORTANT PARAM ####################     
        self.task = 0 # 2 # 0: pick and place, 1: decentralized, 2: collaborative block pushing, 3: Gadi human-robot collaborative pouring
        
        if self.task == 0:   
            self.speed = 200 # mm per second # our default 200 
            self.num_robots = 1
            self.robot1_ip = "192.168.1.219"
            self.robot2_ip = "192.168.1.222"
            self.fixed_orientation = False
            self.do_not_send_commands_to_robot = False
            self.two_dim_motion = False
            self.disable_gripper = False
            self.z_for_2d_motion = 233
            self.z_min = 190 # 200 for diffusion policy pushing
            self.z_min_arm2 = 220 # 200 for diffusion policy pushing
            self.z_min_bound = 170 # double safety! # this should be the most conservative value and lowst
            self.z_max = 650
            self.x_max = 600
            self.y_max = 545
            self.y_max_arm2 = 103
            self.gripper_closing_min_gap = 35 # pot 40 # based on the object information you should test it out first, then verify!!!!!
            self.y_min = -430
            self.x_min = 170.0
            self.arm1_home_position = None
            self.arm1_home_orientation = None
            # self.robot_home_position = [259.1, 2.9, 258.1]  # Default home position [x, y, z]
            self.gripper_closing_distance_threshold = 0.3 # distance threshold for detecting gripper closing between thumb and finger in mocap
            
            self.auto_gripper_reset = False # automatically reset the gripper if it is NOT REACHING THE TARGET closing position for too long
            self.gripper_closing_max_time = 1.0 # max duration for closing gripper
            
        
        if self.task == 1:   
            self.speed = 200 # mm per second # our default 200 
            self.num_robots = 2
            self.robot1_ip = "192.168.1.219"
            self.robot2_ip = "192.168.1.222"
            self.fixed_orientation = False
            self.do_not_send_commands_to_robot = False
            self.two_dim_motion = False
            self.disable_gripper = True
            self.z_for_2d_motion = 233
            self.z_min = 235 # 200 for diffusion policy pushing
            self.z_min_arm2 = 210 # 200 for diffusion policy pushing
            self.z_min_bound = 210 # double safety! # this should be the most conservative value and lowst
            self.z_max = 550
            self.x_max = 600
            self.y_max = 545
            self.y_max_arm2 = 103
            # self.gripper_closing_min_gap = 30 # based on the object information you should test it out first, then verify!!!!!
            self.gripper_closing_min_gap = 25 # based on the object information you should test it out first, then verify!!!!!
            self.y_min = -430
            self.x_min = 170.0
            self.arm1_home_position = None
            self.arm2_home_position = None
            self.arm1_home_orientation = None
            self.arm2_home_orientation = None
            # self.robot_home_position = [259.1, 2.9, 258.1]  # Default home position [x, y, z]
            self.gripper_closing_distance_threshold = 0.11 # distance threshold for detecting gripper closing between thumb and finger in mocap
            self.gripper_closing_max_time = 3.0 # max duration for closing gripper
            
            
            
        if self.task == 2:   
            self.speed = 200 # mm per second # our default 200 
            self.num_robots = 2
            self.robot1_ip = "192.168.1.219"
            self.robot2_ip = "192.168.1.222"
            self.fixed_orientation = True
            self.do_not_send_commands_to_robot = False
            self.two_dim_motion = True
            self.disable_gripper = True
            self.z_for_2d_motion = 209
            self.z_min = 205 # 200 for diffusion policy pushing
            self.z_min_arm2 = 205 # 200 for diffusion policy pushing
            self.z_min_bound = 205 # double safety! # this should be the most conservative value and lowst
            self.z_max = 550
            self.x_max = 1200
            self.y_max = 1200
            self.y_max_arm2 = 1200
            # self.gripper_closing_min_gap = 30 # based on the object information you should test it out first, then verify!!!!!
            self.gripper_closing_min_gap = 25 # based on the object information you should test it out first, then verify!!!!!
            self.y_min = -800
            self.x_min = 0.0
            self.arm1_home_position = None
            self.arm2_home_position = None
            self.downward_quat = np.array([-0.71059049, -0.7035946,   0.00364101,  0.00159283])
            self.arm1_home_orientation = R.from_quat(self.downward_quat)
            self.arm2_home_orientation = R.from_quat(self.downward_quat)
            # self.robot_home_position = [259.1, 2.9, 258.1]  # Default home position [x, y, z]
            self.gripper_closing_distance_threshold = 0.11 # distance threshold for detecting gripper closing between thumb and finger in mocap
            self.gripper_closing_max_time = 3.0 # max duration for closing gripper
            
        if self.task == 3:   
            self.speed = 200 # mm per second # our default 200 
            self.num_robots = 1
            self.robot1_ip = "192.168.1.219"
            self.fixed_orientation = False
            self.do_not_send_commands_to_robot = False
            self.two_dim_motion = False
            self.disable_gripper = True
            self.z_for_2d_motion = 233
            self.z_min = 235 # 200 for diffusion policy pushing
            self.z_min_arm2 = 210 # 200 for diffusion policy pushing
            self.z_min_bound = 210 # double safety! # this should be the most conservative value and lowst
            self.z_max = 550
            self.x_max = 600
            self.y_max = 545
            self.y_max_arm2 = 103
            # self.gripper_closing_min_gap = 30 # based on the object information you should test it out first, then verify!!!!!
            self.gripper_closing_min_gap = 25 # based on the object information you should test it out first, then verify!!!!!
            self.y_min = -430
            self.x_min = 170.0
            self.arm1_home_position = None
            self.arm2_home_position = None
            self.arm1_home_orientation = None
            self.arm2_home_orientation = None
            # self.robot_home_position = [259.1, 2.9, 258.1]  # Default home position [x, y, z]
            self.gripper_closing_distance_threshold = 0.11 # distance threshold for detecting gripper closing between thumb and finger in mocap
            self.gripper_closing_max_time = 3.0 # max duration for closing gripper
            
            

        ##########################################################
       
        # let user confirm number of robots before running
        # input(f"Number of robots is set to {self.num_robots}, press Enter to continue or Ctrl+C to exit")
       
        self.gripper_distance = 800

        self.last_gripper_closing_start = rospy.Time.now()
        self.prev_gripper_open = True
        self.gripper_forced_stop = False

        self.prev_orientation = None
        self.prev_position = None
       
        self.run_sim = run_sim
        # self.downward_orientation = R.from_euler('xyz', [180, 0, 180], degrees=True)

        
        self.servo_path_position_mode = 2 # 0 means servo, 1 means path, and 2 means position

        
        self.arm = self.setup_hw_robot(self.robot1_ip)
        if self.arm is None:
            raise Exception("Robot arm is not set up properly, please check the robot IP")
        if self.num_robots > 1:
            self.arm2 = self.setup_hw_robot(self.robot2_ip)
            if self.arm2 is None:
                raise Exception("Robot arm is not set up properly, please check the robot IP")

        self.horizon = 15
        self.path = []
       
        self.finger_pose = [0, 0, 0]  # Default finger pose [x, y, z]
        self.glove_pose_relative = None
        
        self.glove_2_pose_relative = None
        
        self.thumb_pose = [0, 0, 0]  # Default thumb pose [x, y, z]
        self.target_orientation = [1, 0, 0, 0]  # Default orientation as quaternions

        self.init_glove_pose = None
        self.init_glove_2_pose = None


        # ROS 1 Subscription to the odometry topic
        self.subscription = rospy.Subscriber(
            'vrpn_client_node/glove/pose',
            PoseStamped,
            self.glove_callback
        )
        
        
        
        self.subscription = rospy.Subscriber(
            'vrpn_client_node/glove2/pose',
            PoseStamped,
            self.glove_2_callback
        )

        # ROS 1 Subscription to the odometry topic
        self.subscription = rospy.Subscriber(
            # 'vrpn_client_node/thumb/pose',
            'vrpn_client_node/glove/pose',            
            PoseStamped,
            self.odometry_callback_thumb
        )

        # ROS 1 Subscription to the odometry topic
        self.subscription = rospy.Subscriber(
            'vrpn_client_node/finger/pose',
            PoseStamped,
            self.odometry_callback_finger
        )

        # ROS 1 Publisher for the robot pose
        self.end_effector_pose_publisher = rospy.Publisher(
            'robot_end_effector_pose',
            PoseStamped,
            queue_size=10
        )
        self.end_effector_opening_publisher = rospy.Publisher(
            'robot_end_effector_opening',
            # float type
            Float32,
            queue_size=10
        )
        
        
        
        # ROS 1 Publisher for the robot pose
        self.end_effector_pose_publisher_arm2 = rospy.Publisher(
            'robot_end_effector_pose_arm2',
            PoseStamped,
            queue_size=10
        )
        self.end_effector_opening_publisher_arm2 = rospy.Publisher( 
            'robot_end_effector_opening_arm2',
            # float type
            Float32,
            queue_size=10
        )

    def setup_hw_robot(self, ip):
        if ip is not None:
            print(f"Connecting to robot arm at IP: {ip}")
            arm = XArmAPI(ip)
            if not self.disable_gripper:
                # Set the gripper mode (0: Position Mode, 1: Speed Mode)
                arm.set_gripper_mode(0)
                arm.set_gripper_speed(2000)
                arm.set_gripper_enable(True)
                # open the gripper
                # arm.set_gripper_position(800, wait=False)
            
            # input('Press Enter to enable robot arm motion!')
            arm.motion_enable(enable=True)
            # input('Press Enter to set mode and state to 0')

            if self.servo_path_position_mode == 0:
                arm.set_mode(1)  # Enable servo mode
                input('Using servo mode. No collision checks! Press Enter to confirm and continue')
                input('Double check: Using servo mode. No collision checks! Press Enter AGAIN to confirm and continue')
            else:
                arm.set_mode(0)

            arm.set_pause_time(0.0)

            arm.set_state(state=0)
            # # arm.move_gohome(wait=True)
            # if self.run_sim == False:
            #     input('About to run real hardware experiment. Press Enter if you are ready!')
            return arm
        else:
            rospy.logerr(f"Robot IP is not set, please set the robot IP, IP is: {ip}")
            return None
        
            
        
    def odometry_callback_finger(self, msg):
        if self.disable_gripper:
            return
        """Callback function to update target pose based on odometry."""
        self.finger_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]

    def odometry_callback_thumb(self, msg):
        if self.disable_gripper:
            return
        """Callback function to update target pose based on odometry."""
        self.thumb_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        
        # check the distance between thumb and target if target is not 0, 0, 0
        if self.finger_pose != [0, 0, 0]:
            # calculate the distance between thumb and target
            distance_sq = np.sum(np.square(np.array(self.thumb_pose) - np.array(self.finger_pose)))

            gripper_delta = 7 # 850 full range from close to open, we made it 800 to avoid current spike
            if distance_sq < self.gripper_closing_distance_threshold ** 2:
                self.gripper_distance = max(self.gripper_closing_min_gap, self.gripper_distance - gripper_delta)
            else:
                self.gripper_distance = min(800, self.gripper_distance + gripper_delta)

    def glove_callback(self, msg):
        """Callback function to update target pose based on odometry."""


        if self.init_glove_pose is None:
            self.init_glove_pose = msg.pose
            rospy.loginfo(f"Initial glove pose: {self.init_glove_pose}")
        else:
            self.glove_pose_relative = self.calculate_relative_pose(msg.pose, self.init_glove_pose, self.arm)
            # rospy.loginfo(f"Updated glove position: {self.glove_pose_relative}")
            
    
    def glove_2_callback(self, msg):
        # if single arm, skip this
        if self.num_robots < 2:
            # rospy.logwarn(f"Skipping glove 2 callback since only one robot is used")
            return
        """Callback function to update target pose based on odometry."""
        if self.init_glove_2_pose is None:
            self.init_glove_2_pose = msg.pose
            rospy.loginfo(f"Initial glove 2 pose: {self.init_glove_2_pose}")
        else:
            self.glove_2_pose_relative = self.calculate_relative_pose(msg.pose, self.init_glove_2_pose, self.arm2)
            rospy.loginfo(f"Updated glove 2 position: {self.glove_2_pose_relative}")

    def publish_robot_pose(self, current_arm, end_effector_pose_publisher, end_effector_opening_publisher):
        """Publish the robot pose."""

        # get current end effector pose
        # NO LONGER USED! USED AXIS ANGLE INSTEAD TO AVOID SINGULARITY (GIMBAL LOCK)
        # pose = self.arm.get_position(is_radian=True)
        code, position = current_arm.get_position_aa(is_radian=True)
        
        x, y, z, rx, ry, rz = position
        
        # assert rx ry rz should be using radians
        assert abs(rx) <= 2*np.pi, "Roll angle is not within the range of radians"
        assert abs(ry) <= 2*np.pi, "Pitch angle is not within the range of radians"
        assert abs(rz) <= 2*np.pi, "Yaw angle is not within the range of radians"
        
        # convert from axis angle to quaternion
        cur_quaternion = R.from_rotvec([rx, ry, rz]).as_quat()
        # quaternion is in form of [x, y, z, w]

        if not self.disable_gripper:
            gripper_status_and_pos = current_arm.get_gripper_position()
        else:
            gripper_status_and_pos = [0, 0]
        gripper_opening = gripper_status_and_pos[1]
        # convert to PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "robot_base"
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        # convert roll pitch yaw to quaternion
        # sanity check to ensure the euler angles are within the range radians
        
        # rospy.loginfo_throttle(1.0,"current euler angles (should be in rad NOT degrees!): {pose[1][3:]}")
        # assert  np.all(np.abs(pose[1][3:]) <= 2 * np.pi), "Euler angles are not within the range of radians"
        # quaternion = R.from_euler('xyz', pose[1][3:]).as_quat()
        pose_stamped.pose.orientation.x = cur_quaternion[0]
        pose_stamped.pose.orientation.y = cur_quaternion[1]
        pose_stamped.pose.orientation.z = cur_quaternion[2]
        pose_stamped.pose.orientation.w = cur_quaternion[3]
        # publish the pose
        end_effector_pose_publisher.publish(pose_stamped)
        # publish the gripper opening
        end_effector_opening_publisher.publish(gripper_opening)


    def update_robot_pose(self, current_arm, glove_pose_relative):
        """Update robot joints based on IK solutions."""
        if glove_pose_relative is not None:
            # convert the glove pose orientation to euler angles
            glove_pose_R = R.from_quat([
                glove_pose_relative.orientation.x,
                glove_pose_relative.orientation.y,
                glove_pose_relative.orientation.z,
                glove_pose_relative.orientation.w
            ])
            euler_angles = glove_pose_R.as_euler('xyz', degrees=True)

            rospy.loginfo_throttle(1.0, f"Current commanded position: {glove_pose_relative.position.x, glove_pose_relative.position.y, glove_pose_relative.position.z}")
            
            # get position, compensate by adding the home position, and convert to mm
            if current_arm == self.arm:
                if self.arm1_home_position is None:
                    # self.arm1_home_position = current_arm.get_position(is_radian=False)[0]
                    code, position = current_arm.get_position_aa(is_radian=True)
                    x, y, z, rx, ry, rz = position
                    self.arm1_home_position = [x, y, z]
                    rospy.loginfo(f"Initial arm 1 home position: {self.arm1_home_position}")
                cur_arm_home_position = self.arm1_home_position            
            elif current_arm == self.arm2:
                if self.arm2_home_position is None:
                    code, position = current_arm.get_position_aa(is_radian=True)
                    x, y, z, rx, ry, rz = position
                    self.arm2_home_position = [x, y, z]
                    rospy.loginfo(f"Initial arm 2 home position: {self.arm2_home_position}")
                cur_arm_home_position = self.arm2_home_position
            x_pos = glove_pose_relative.position.x * 1000 + cur_arm_home_position[0]
            y_pos = glove_pose_relative.position.y * 1000 + cur_arm_home_position[1]
            z_pos = glove_pose_relative.position.z * 1000 + cur_arm_home_position[2]
            # rospy.loginfo_throttle(1.0, f"compensating for home position: {x_pos, y_pos, z_pos}, and making the initial orientation to be pointing down")
            
            x_pos, y_pos, z_pos = self.check_boundaries(x_pos, y_pos, z_pos, current_arm)

            if self.do_not_send_commands_to_robot:
                rospy.logwarn(f"Not sending commands to robot")
                return

            if not self.disable_gripper:
                # gripper stuff currently only for arm 1
                if current_arm == self.arm:
                    if self.gripper_distance > 425 and self.prev_gripper_open == False:
                        current_arm.set_gripper_position(800, wait=False)
                        self.prev_gripper_open = True
                    elif self.gripper_distance <= 425 and self.prev_gripper_open == True:
                        self.last_gripper_closing_start = rospy.Time.now()
                        current_arm.set_gripper_position(self.gripper_closing_min_gap, wait=False)
                        self.prev_gripper_open = False
                        self.gripper_forced_stop = False

                    if (self.prev_gripper_open == False) and self.auto_gripper_reset:
                        if (rospy.Time.now() - self.last_gripper_closing_start > rospy.Duration(self.gripper_closing_max_time)) and self.gripper_forced_stop == False:
                            rospy.logwarn(f"Closing gripper for more than {self.gripper_closing_max_time} seconds, stopping the gripper movement to avoid time delay")
                            # set the gripper to where it is now
                            cur_gripper_position = current_arm.get_gripper_position()[1]
                            # expand a bit to avoid big current
                            rospy.logwarn(f"Stopping gripper at position: {cur_gripper_position}")
                            current_arm.set_gripper_position(cur_gripper_position, wait=False)
                            self.gripper_forced_stop = True
                        # self.last_gripper_closing_start = rospy.Time.now()


            if self.servo_path_position_mode == 0:
                raise Exception("set_servo_cartesian is not supported yet")
                # # set_servo_cartesian 
                #     TODO: potentially unsafe, not using this for now
                # current_arm.set_servo_cartesian()
                


            elif self.servo_path_position_mode == 1:
                raise Exception("Path mode is not supported now yet")
                # arc_radius = 0.5
                # if len(self.path) > self.horizon:
                #     current_arm.move_arc_lines(self.path, is_radian=False, times=1, first_pause_time=0.000001, repeat_pause_time=0, automatic_calibration=True, speed=self.speed, mvacc=1000, mvtime=None, wait=False)
                #     # pop half of the path
                #     self.path = []
                # else:
                #     self.path.append([x_pos, y_pos, z_pos, euler_angles[0], euler_angles[1], euler_angles[2], arc_radius])

            elif self.servo_path_position_mode == 2:
                # update the target position
                # current_arm.set_pause_time(0.000000001)
                status = current_arm.get_state()  # Check if robot is ready

                # skip if the robot is in motion
                if status[1] == 1:
                    rospy.logwarn_throttle(1.0, f"Robot is in motion, skipping this iteration")
                    return
                else:
                    current_arm.set_position(
                        x=x_pos,
                        y=y_pos,
                        z=z_pos,
                        roll=euler_angles[0],
                        pitch=euler_angles[1],
                        yaw=euler_angles[2],
                        speed=self.speed,
                        mvacc=3000,
                        radius=0.0,
                        wait=False,
                        is_radian=False,
                        relative=False,
                        motion_type=0
                    )
                # time.sleep(1.0/refresh_rate)
            else:
                rospy.logerr(f"Invalid mode selected")


    def check_boundaries(self, x_pos, y_pos, z_pos, current_arm):
        # check if Z is less than threshold, and print error that it is going to hit table 
        if self.two_dim_motion:
            z_pos = self.z_for_2d_motion 
            if z_pos < self.z_min_bound:
                rospy.logerr(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                rospy.logerr(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                rospy.logerr(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                rospy.logerr(f"DOUBLE CHECK THIS: YOUR Z position min bound is set too low, which is: {z_pos}, setting it to : {self.z_min_bound}, otherwise it will hit the table")
                z_pos = self.z_min_bound
            rospy.loginfo(f"2D motion enabled")
        else:
            if current_arm == self.arm:
                cur_z_min = self.z_min
            else:
                cur_z_min = self.z_min_arm2
            if z_pos < cur_z_min:
                # rospy.logerr(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                rospy.logerr(f"Z position is too low, which is: {z_pos}, setting it to : {cur_z_min}, otherwise it will hit the table")
                z_pos = cur_z_min
                if z_pos < self.z_min_bound:
                    rospy.logerr(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    rospy.logerr(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    rospy.logerr(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    rospy.logerr(f"DOUBLE CHECK THIS: YOUR Z position min bound is set too low, which is: {z_pos}, setting it to : {self.z_min_bound}, otherwise it will hit the table")
                    z_pos = self.z_min_bound
            if z_pos > self.z_max:
                
                # rospy.logwarn(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # print following the same warning format as above
                rospy.logwarn(f"Z position is too high, which is: {z_pos}, setting it to : {self.z_max}, otherwise it will go out of safe workspace")
                z_pos = self.z_max
        
        if x_pos > self.x_max:
            # rospy.logwarn(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # following the same warning format as above
            rospy.logwarn(f"X position is too high, which is: {x_pos}, setting it to : {self.x_max}, otherwise it will go out of safe workspace")
            x_pos = self.x_max
        if x_pos < self.x_min:
            # rospy.logwarn(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # following the same warning format as above
            rospy.logwarn(f"X position is too low, which is: {x_pos}, setting it to : {self.x_min}, otherwise it will go out of safe workspace")
            x_pos = self.x_min

        if current_arm == self.arm:
            y_max = self.y_max
        else:
            y_max = self.y_max_arm2
        if y_pos > y_max:
            # rospy.logwarn(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # following the same warning format as above
            rospy.logwarn(f"Y position is too high, which is: {y_pos}, setting it to : {y_max}, otherwise it will go out of safe workspace")
            y_pos = y_max
            
        if y_pos < self.y_min:
            # rospy.logwarn(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # following the same warning format as above
            rospy.logwarn(f"Y position is too low, which is: {y_pos}, setting it to : {self.y_min}, otherwise it will go out of safe workspace")
            y_pos = self.y_min
        return x_pos, y_pos, z_pos

    def calculate_relative_pose(self, current_pose, initial_glove_pose, current_arm):
        robot_init_orientation = None
        if current_arm == self.arm:
            if self.arm1_home_orientation is None:
                code, position = current_arm.get_position_aa(is_radian=True)
                x, y, z, rx, ry, rz = position
                self.arm1_home_orientation = R.from_rotvec([rx, ry, rz])
            # convert from axis angle to Rotation
            robot_init_orientation = self.arm1_home_orientation
            
        
        elif current_arm == self.arm2:
            if self.arm2_home_orientation is None:
                # self.arm2_home_orientation = current_arm.get_position(is_radian=False)[1][3:]
                code, position = current_arm.get_position_aa(is_radian=True)
                x, y, z, rx, ry, rz = position
                self.arm2_home_orientation = R.from_rotvec([rx, ry, rz])
            robot_init_orientation = self.arm2_home_orientation
            
        else:
            raise Exception("Invalid arm selected")
                
        # raise Exception("Invalid arm selected")
        
        
        # make it pose not posestamped
        relative_pose = Pose()
        relative_position = np.array([
            current_pose.position.x - initial_glove_pose.position.x,
            current_pose.position.y - initial_glove_pose.position.y,
            current_pose.position.z - initial_glove_pose.position.z,
        ])
        if self.fixed_orientation:
            relative_orientation = robot_init_orientation.as_quat()
            # relative_orientation = self.downward_quat
            rospy.loginfo_once(f"Currently using fixed, hard coded relative orientation: {relative_orientation}")
            # relative_orientation = self.downward_orientation.as_quat()
            rospy.loginfo_once(f"Currently using fixed, hard coded relative orientation: {relative_orientation}")
        else:
            # Compute inverse of the initial orientation only once
            # if initial_glove_pose and not hasattr(self, 'init_glove_pose_quat_inv'):
            initial_glove_pose_quat_inv = R.from_quat([
                initial_glove_pose.orientation.x,
                initial_glove_pose.orientation.y,
                initial_glove_pose.orientation.z,
                initial_glove_pose.orientation.w
            ]).inv()

            # Compute current orientation quaternion
            R_current = R.from_quat([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ])

            # Compute the relative orientation efficiently
            R_intial2current = R_current * initial_glove_pose_quat_inv

            # Apply downward compensation in the correct order
            R_intial2current_compensated = R_intial2current * robot_init_orientation

            # Convert to quaternion
            relative_orientation = R_intial2current_compensated.as_quat()


            # # get the relative orientation between the relative orientation and the downward orientation
            # diff_from_downward = robot_init_orientation.inv() * R.from_quat(relative_orientation)
            # diff_euler = diff_from_downward.as_euler('xyz', degrees=True)
            # # if diff_euler[0] > 5 or diff_euler[1] > 5 or diff_euler[2] > 5:
            # #     rospy.logwarn_throttle(1.0,f"Current orientation is far from downward orientation, difference is: {diff_euler}")
            # # else:
            # #     rospy.loginfo_throttle(1.0, f"Current orientation is close to downward orientation, difference is: {diff_euler}")
               
        relative_pose.position.x = relative_position[0]
        relative_pose.position.y = relative_position[1]
        relative_pose.position.z = relative_position[2]
        relative_pose.orientation.x = relative_orientation[0]
        relative_pose.orientation.y = relative_orientation[1]
        relative_pose.orientation.z = relative_orientation[2]
        relative_pose.orientation.w = relative_orientation[3]
        return relative_pose

        
def main():
    # Initialize ROS 1
    rospy.init_node('xarm_controller', anonymous=True)
    # Initialize the xArm simulator
    xarm_controller = XArmController()
    
    refresh_rate = 120
    # Simulation loop
    rate = rospy.Rate(refresh_rate)  # 60 Hz
    # try:
    while not rospy.is_shutdown():# and p.isConnected():
        xarm_controller.publish_robot_pose(xarm_controller.arm, xarm_controller.end_effector_pose_publisher, xarm_controller.end_effector_opening_publisher)
        xarm_controller.update_robot_pose(xarm_controller.arm, xarm_controller.glove_pose_relative)
        if xarm_controller.num_robots > 1:
            rospy.loginfo_throttle(1.0, "Updating robot 2")
            xarm_controller.publish_robot_pose(xarm_controller.arm2, xarm_controller.end_effector_pose_publisher_arm2, xarm_controller.end_effector_opening_publisher_arm2)
            xarm_controller.update_robot_pose(xarm_controller.arm2, xarm_controller.glove_2_pose_relative)
            
        rate.sleep()  # Maintain loop rate

if __name__ == '__main__':
    main()