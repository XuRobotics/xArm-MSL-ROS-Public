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
import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import example.wrapper.xarm6.xarm_sim as xarm_sim
# import geometry_msgs/PoseStamped
from geometry_msgs.msg import PoseStamped
from pathlib import Path
from sak.URDFutils import URDFutils
from pydrake.math import RotationMatrix

from pydrake.all import *
import pydrake.multibody.plant as pmp


class XArmController:
    def __init__(self, bullet_client, xarm_sim, offset):

    # def __init__(self):
        self.bullet_client = bullet_client
        self.xarm_sim = xarm_sim
        self.offset = offset
        self.end_effector_index = 6  # xArm end-effector link index
        self.target_pose = [0, 0, 0]  # Default target pose [x, y, z]
        self.finger_pose = [0, 0, 0]  # Default finger pose [x, y, z]
        self.thumb_pose = [0, 0, 0]  # Default thumb pose [x, y, z]
        self.target_orientation = [1, 0, 0, 0]  # Default orientation as quaternions
        self.package_path = "/home/sam/xArm-MSL-ROS/xArm-Python-SDK/example/wrapper/xarm6/xarm_description"
        self.package_name = "xarm6/"
        self.urdf_name = "xarm6_with_push_gripper.urdf"
        urdf_utils = URDFutils(self.package_path, self.package_name, self.urdf_name)
        urdf_utils.modify_meshes()
        urdf_utils.remove_collisions_except([])
        urdf_utils.add_actuation_tags()
        self.urdf_str, self.temp_urdf = urdf_utils.get_modified_urdf()
        self.moving_average_joint_positions = [0, 0, 0, 0, 0, 0]
        self.initialized = False
        moving_horizon = 5
        self.alpha = 1.0 / moving_horizon
        self.joint_position_jump_threshold = np.deg2rad(30)

        self.failed_attempts = 0



        # ROS 1 Subscription to the odometry topic
        self.subscription = rospy.Subscriber(
            'vrpn_client_node/drone1/pose',
            PoseStamped,
            self.odometry_callback
        )


        # ROS 1 Subscription to the odometry topic
        self.subscription = rospy.Subscriber(
            'vrpn_client_node/thumb/pose',
            PoseStamped,
            self.odometry_callback_thumb
        )




        # ROS 1 Subscription to the odometry topic
        self.subscription = rospy.Subscriber(
            'vrpn_client_node/finger/pose',
            PoseStamped,
            self.odometry_callback_finger
        )


    def odometry_callback_finger(self, msg):
        """Callback function to update target pose based on odometry."""
        self.finger_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]

    def odometry_callback_thumb(self, msg):
        """Callback function to update target pose based on odometry."""
        self.thumb_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        

        # check the distance between thumb and target if target is not 0, 0, 0
        if self.finger_pose != [0, 0, 0]:
            # calculate the distance between thumb and target
            distance = np.linalg.norm(np.array(self.thumb_pose) - np.array(self.finger_pose))
            # if distance is less than 0.1, update the target pose to thumb pose
            rospy.logwarn(f"Distance between thumb and target: {distance}")
            if distance < 0.05:
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")
                rospy.logerr(f"gripper closing detected!!!!!")

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


        # origin:     
#        [0.3172536790370941, -0.3267155587673187, 0.8736760020256042]

        # substract target pose with the origin
        self.target_pose = [
            self.target_pose[0] - 0.3172536790370941 + 0.2,
            self.target_pose[1] + 0.3267155587673187 + 0.2,
            self.target_pose[2] - 0.8736760020256042 + 0.2,
        ]


        # Extract orientation as quaternion from odometry
        orientation = msg.pose.orientation
        # # set orientation as pointing down
        

        self.target_orientation = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        # print(f"Updated target pose: {self.target_pose}")
        rospy.loginfo(f"Updated target pose: {self.target_pose}")
        # print the current robot end effector pose
        # print("Current end effector pose: ", self.bullet_client.getLinkState(self.xarm_sim.xarm, self.end_effector_index)[0])


    def solve_inverse_kinematics(self):
        """Solve inverse kinematics for the target pose and orientation using Drake."""
        # Setup the Drake plant and context for IK solving
        builder = DiagramBuilder()
        plant, scene_graph = pmp.AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
        parser = Parser(plant, scene_graph)
        abs_path = Path(self.package_path).resolve().__str__()
        parser.package_map().Add(self.package_name.split("/")[0], abs_path + "/" + self.package_name)
        model = parser.AddModels(self.temp_urdf.name)[0]
        plant.Finalize()

        # Initialize context
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        # Get the end-effector link and its frame
        eef_link_name = "link6"  # Adjust as needed
        eef = plant.GetBodyByName(eef_link_name)
        eef_frame = eef.body_frame()

        # Create the desired pose using target position and orientation
        target_transform = RigidTransform(
            # RollPitchYaw(0, 0, 0),
            Quaternion(self.target_orientation),  # Target orientation
            self.target_pose,  # Target position
        )

        # Inverse Kinematics setup
        ik = InverseKinematics(plant, plant_context, with_joint_limits=True)

        # # Add position and orientation constraints
        # ik.AddPositionCost(
        #     eef_frame, [0, 0, 0], plant.world_frame(), target_transform.translation(), np.eye(3)
        # )

        # add position constraint using the info above and API below:
#         AddPositionConstraint(self: pydrake.multibody.inverse_kinematics.InverseKinematics, frameB: pydrake.multibody.tree.Frame, p_BQ: numpy.ndarray[numpy.float64[3, 1]], frameAbar: pydrake.multibody.tree.Frame, X_AbarA: Optional[pydrake.math.RigidTransform], p_AQ_lower: numpy.ndarray[numpy.float64[3, 1]], p_AQ_upper: numpy.ndarray[numpy.float64[3, 1]]) -> pydrake.solvers.Binding[Constraint]

        # specify each input argument
        ik.AddPositionConstraint(
            frameB=eef_frame,
            p_BQ=[0, 0, 0],  # position of the end effector in the end effector frame
            frameAbar=plant.world_frame(),
            X_AbarA=None,
            p_AQ_lower=target_transform.translation(),
            p_AQ_upper=target_transform.translation(),
        )
        rospy.loginfo(f"Target position: {target_transform.translation()}")


        # print types of  eef_frame RotationMatrix plant target_transform
        # print(type(eef_frame))
        # print(type(RotationMatrix()))
        # print(type(plant.world_frame()))
        # print(type(target_transform.rotation()))
        # print out their values
        # print(eef_frame)
        # print(RotationMatrix())
        # print(plant.world_frame())
        # print(target_transform.rotation())
        # convert rotation matrix into pydrake.multibody.tree.Frame,
        # time.sleep(10)

        # ik.AddOrientationConstraint(
        #     eef_frame,
        #     RotationMatrix(),
        #     plant.world_frame(),
        #     RotationMatrix(),
        #     # target_transform.rotation(),
        #     angle_bound=float(0.01),  # Orientation tolerance
        # )

        # current orientation 0.16  0.12 -3.01for roll pitch yaw
        # desired orientation is 0 0 0
        # add a compensation to rotate current to desired
        
        # first get desired from -0.11 -0.18 -2.77 
        desired_orientation = RotationMatrix(RollPitchYaw(np.pi, -np.pi/2, 0))
        # desired_orientation = RotationMatrix(RollPitchYaw(np.pi, -np.pi/2, 0))
        # then get the current orientation
        offset_orientation = RotationMatrix(RollPitchYaw(0.16, 0.12, -3.01))
        offset_orientation_inv = offset_orientation.inverse()
        compensation_orientation = offset_orientation_inv.multiply(desired_orientation)

        # multiply the current orientation with the compensation orientation
        target_orientation = target_transform.rotation().multiply(compensation_orientation)
        # update the target transform with the new orientation
        target_transform.set_rotation(target_orientation)



        # Add orientation constraint
        ik.AddOrientationConstraint(
            frameAbar=eef_frame,
            R_AbarA=RotationMatrix(),  # Identity rotation
            frameBbar=plant.world_frame(),
            R_BbarB= target_transform.rotation(),  # RotationMatrix from target_transform
            theta_bound=0.3,
        )
        # print the rotation matrix in roll pitch yaw, keep only 2 decimal points
        rpy = target_transform.rotation().ToRollPitchYaw()
        rpy_vect = rpy.vector()
        rospy.loginfo(f"Target orientation in Roll Pitch Yaw: {np.round(rpy_vect, 2)}")
        

        # Setup the optimization problem
        prog = ik.get_mutable_prog()
        q = ik.q()  # Decision variable: joint positions
        q0 = plant.GetPositions(plant_context)  # Initial guess for joint angles
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)

        # Solve the IK problem
        result = Solve(ik.prog())
        if not result.is_success():
            # raise ValueError("Inverse Kinematics solution not found.")
            print("Inverse Kinematics solution NOT found!!!!")
            print("Inverse Kinematics solution NOT found!!!!")
            print("Inverse Kinematics solution NOT found!!!!")
            return None

        # Extract the joint angle solution
        joint_angles = result.GetSolution(q)
        # log throttle 1hz 
        rospy.loginfo(f"Joint angles: {joint_angles}")  
        

        # convert angles into degrees
        # joint_angles = np.rad2deg(joint_angles)

        return joint_angles.tolist()

    def update_robot_pose(self):
        """Update robot joints based on IK solutions."""
        joint_angles = self.solve_inverse_kinematics()
        if joint_angles is None:
            return
        
        if sum(self.moving_average_joint_positions) == 0:
            self.moving_average_joint_positions = joint_angles
            rospy.logwarn("initialization, show only show once")
            return
        else:
            # skip if any of the joint angles jump more than the threshold
            if any(np.abs(np.array(joint_angles) - np.array(self.moving_average_joint_positions)) > self.joint_position_jump_threshold):
                self.failed_attempts += 1
                rospy.logerr("Joint angles jump detected. Skip updating joint angles.")
                rospy.logerr("Joint angles jump detected. Skip updating joint angles.")
                rospy.logerr("Joint angles jump detected. Skip updating joint angles.")
                # print the difference in degrees
                rospy.logerr(f"Joint angles jump detected. Skip updating joint angles. Difference: {np.rad2deg(np.abs(np.array(joint_angles) - np.array(self.moving_average_joint_positions)) )}")
                if self.failed_attempts > 5 & self.initialized == False:
                    rospy.logerr("Too many failed attempts. Reset moving average joint positions.")
                    self.moving_average_joint_positions = joint_angles
                    self.failed_attempts = 0               
                return
            else:
                if self.initialized == False:
                    self.initialized = True
                self.failed_attempts = 0    
                self.moving_average_joint_positions = self.alpha * np.array(joint_angles) + (1 - self.alpha) * np.array(self.moving_average_joint_positions)
                print(f"Joint angles within limit. Moving average joint positions: {self.moving_average_joint_positions}")
            
    
                
        rospy.logerr(f"Joint angles: {joint_angles}")
        xarmNumDofs = 6
        for i in range(xarmNumDofs):
            self.bullet_client.setJointMotorControl2(
                bodyUniqueId=self.xarm_sim.xarm,
                jointIndex=i+1,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=5 * 240,
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