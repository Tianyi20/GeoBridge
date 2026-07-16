from AssemblySim import AssemblySim
import pybullet as p
import os
from episode_writer import EpisodeWriter
import json
from pathlib import Path
from icecream import ic
import random
import pybullet_data as pd
import math
import time
import numpy as np
import cv2
p.connect(p.GUI)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.setAdditionalSearchPath(pd.getDataPath())
timeStep=1./120.
p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

Sim = AssemblySim(p, offset=[0, 0, 0], control_dt = timeStep, seed = 812553,
                randomize_initial_ee_pose= True)

Sim.make_scene(
    env_mesh_path= "./data/background/repaired_table/tabletop.obj",
    assembly_parent_path= "./data/objects/assembly/tool/tool.obj",
    assembly_parent_collision_path = None,
    assembly_child_path = "data/objects/assembly/child/assembly_child.obj",
    initial_grasp_path = "data/objects/assembly/tool/tool_grasp.yaml",
    parentobj_pose_base = [0.45, -0.1, 0.015],
    parentobj_euler_base = [0.0, 0.0, -np.pi/2.0],
    childobj_pose_base = [0.7, -0.1, 0.015],
    childobj_euler_base = [0.0, 0.0, -np.pi/2.0],
    # child local +Z is the insertion axis; parent OBJ origin is ring center
    assembly_axis_local = [0.0, 0.0, 1.0],
    child_to_parent_target_pos = [0.0, 0.0, 0.0],
    child_to_parent_target_euler = [0.0, 0.0, 0.0],
    safe_assembly_approach = 0.12,
    randomize_lighting= True,
    # outlier scene         
    randomize_outlscene  = True,
    outlscene_xyz_jit    = 0.015,
    outlscene_eul_jit    = 0.001,
    # plane height randomization
    randomize_plane_height = True,
    plane_height_jit = 0.003,
    randomize_parent_objpose  = True,
    parentobj_x_jit    = 0.1,
    parentobj_y_jit    = 0.15,
    parentobj_z_jit    = 0.0,
    parentobj_z_eul_jit = np.pi/3,
    randomize_child_objpose  = True,
    childobj_x_jit    = 0.1,
    childobj_y_jit    = 0.15,
    childobj_z_jit    = 0.0,
    childobj_z_eul_jit = np.pi/18,
    randomize_campose = True,
    cam_xyz_jit  = 0.01,
    cam_eul_jit  = 0.005,
    randomize_fisheye_cam = True,
    fisheye_eyz_jit = 0.005,
    fisheye_eul_jit = 0.002,
    randomize_camera_intrinsic = True,
    randomize_image_noise= True,
    randomize_object_color = True,
    randomize_robot_texture = True,
    robot_texture_patterns= ("checkers", "gradient", "noise", "plain"),
    object_color_mode = "bounded",  # "bounded" or "recolor"
    object_color_strength = 0.5,
    randomize_distractors= False,
    distractor_root= "/mnt/storage/GoogleScannedObjects",
    distractor_num_range= (0, 5),
    distractor_target_size_range= (0.1, 0.5),
    distractor_workspace = ((-0.2, 1.3), (-0.72, 0.42)),
    distractor_clearance = 0.06,
    distractor_path_clearance = 0.06,
    # at least 10 pixel of the target object
    distractor_min_target_mask_pixels= 10,
    )

Sim.enable_high_quality_rendering()

sim_step = 0
timestamp = 0.0
record_idx = 0
record_every_n_sim_steps = 12

# while True:
#     p.stepSimulation()

try:
    while (not Sim.done):
        p.stepSimulation()
        Sim.step()
        sim_step += 1
        time.sleep(0.001)
        if sim_step % record_every_n_sim_steps == 0:
            # RGB = Sim.get_eye_in_hand_image()
            # RGB_agent = Sim.direct_get_agent_view()

            # cv2.imwrite("temp_fisheye.png", cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("temp_agentview.png", cv2.cvtColor(RGB_agent, cv2.COLOR_RGBA2BGRA))
            # obs = Sim.collect_observation(direct= True,
            #                               use_eye_in_hand= True)

            record_idx += 1
            # print(obs["robot0_gripper_qpos"])
            print(record_idx)
        # time.sleep(0.05)
            print(Sim.is_success(debug= False))

        #Sim.collect_action()

except KeyboardInterrupt:
    print("Stopped by user")
