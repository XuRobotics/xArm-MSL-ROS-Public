# %%
import pydrake.multibody.plant as pmp
from sak.URDFutils import URDFutils
from sak.quickload2drake import add_ground_with_friction, robot_joint_teleop
from pathlib import Path
import logging
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    Parser,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    InverseKinematics,
    Solve,
    AddDefaultVisualization,
)
import time
from xarm.wrapper import XArmAPI
import math
# %%
meshcat = StartMeshcat()
# %%
# fix abs path!!!
package_path = "/home/sam/xArm-MSL-ROS/xArm-Python-SDK/example/wrapper/xarm6/xarm_description"
package_name = "xarm6/"
urdf_name = "xarm6_with_push_gripper.urdf"
urdf_utils = URDFutils(package_path, package_name, urdf_name)
urdf_utils.modify_meshes()
urdf_utils.remove_collisions_except([])
urdf_utils.add_actuation_tags()
urdf_str, temp_urdf = urdf_utils.get_modified_urdf()
# %%
# joint angle direction signs are verified to be the same between the hardware and the robot sim
def setup_hw_robot(ip = "192.168.1.222"):
    if ip is not None:
        arm = XArmAPI(ip)
        input('Press Enter to enable motion')
        arm.motion_enable(enable=True)
        input('Press Enter to set mode and state to 0')
        arm.set_mode(0)
        arm.set_state(state=0)
        input('Press Enter to go home')
        arm.move_gohome(wait=True)
        return arm
    else:
        print('No hardware arm found')
        return None
def eef_ik_teleop(meshcat, package_path, package_name, temp_urdf, arm=None):
    builder = DiagramBuilder()
    plant, scene_graph = pmp.AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant, scene_graph)
    abs_path = Path(package_path).resolve().__str__()
    parser.package_map().Add(package_name.split("/")[0], abs_path + "/" + package_name)
    print(package_name.split("/")[0])
    model = parser.AddModels(temp_urdf.name)[0]
    plant.Finalize()
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    # test without the gripper
    eef_link_name = "link6" # "push_gripper_base_link"
    eef = plant.GetBodyByName(eef_link_name)
    eef_frame = eef.body_frame()
    eef_home_pose = plant.EvalBodyPoseInWorld(
        plant_context, plant.GetBodyByName(eef_link_name)
    )
    pose_slider_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    init_vals = np.append(
        eef_home_pose.translation(), eef_home_pose.rotation().ToRollPitchYaw().vector()
    )
    for ii in range(len(pose_slider_names)):
        meshcat.AddSlider(
            pose_slider_names[ii],
            min=-np.pi,
            max=np.pi,
            step=5e-3,
            value=init_vals[ii],
        )
    meshcat.AddButton("Send to HW")
    while True:
        pose = RigidTransform(
            RollPitchYaw(
                meshcat.GetSliderValue("roll"),
                meshcat.GetSliderValue("pitch"),
                meshcat.GetSliderValue("yaw"),
            ),
            [
                meshcat.GetSliderValue("x"),
                meshcat.GetSliderValue("y"),
                meshcat.GetSliderValue("z"),
            ],
        )
        ik = InverseKinematics(plant, plant_context, with_joint_limits=True)
        print("Pose: ", pose.translation(), pose.rotation().ToRollPitchYaw().vector())
        ik.AddPositionCost(
            eef_frame,
            [0, 0, 0],
            plant.world_frame(),
            pose.translation(),
            np.eye(3),
        )
        # add the orientation constraint
        # print types of  eef_frame RotationMatrix plant target_transform
        print(type(eef_frame))
        print(type(RotationMatrix()))
        print(type(plant.world_frame()))
        print(type(pose.rotation()))
        # print out their values
        print(eef_frame)
        print(RotationMatrix())
        print(plant.world_frame())
        print(pose.rotation())
        
        ik.AddOrientationConstraint(
            eef_frame,
            RotationMatrix(),
            plant.world_frame(),
            pose.rotation(),
            1e-2,
        )
        prog = ik.get_mutable_prog()
        q = ik.q()
        q0 = plant.GetPositions(plant_context)
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        result = Solve(ik.prog())
        q_res, flag = result.GetSolution(q), result.is_success()
        plant.SetPositions(plant_context, q_res)
        diagram.ForcedPublish(context)
        print("IK success: ", flag)
        print("IK solution: ", q_res)
        if meshcat.GetButtonClicks("Send to HW"):
            if arm == None:
                print('No hardware arm found')
                time.sleep(2)
            else:
                print('sending command to hardware')
                print(q_res)
                speed = math.radians(10)
                input(f'Press Enter to send command at speed {speed}')
                arm.set_servo_angle(
                    angle=q_res, speed=speed, is_radian=True, wait=True
                )
                print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
                print(q_res)
                time.sleep(5)
            meshcat.AddButton("Send to HW")
# %%
arm = setup_hw_robot(ip=None)
print(arm)
input('Press Enter to start teleop')
eef_ik_teleop(meshcat, package_path, package_name, temp_urdf, arm=arm)
