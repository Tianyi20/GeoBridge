from WrenchSim import WrenchSim
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

Sim = WrenchSim(p, offset=[0, 0, 0], control_dt = timeStep, seed = 32248)

Sim.make_scene(
    env_mesh_path= "./data/background/repaired_table/tabletop.obj",
    manipulated_obj_path= "./data/objects/screw/screw.obj",
    manipulated_obj_collision_path = "./data/objects/screw/screw_collision_asset.obj",
    clipper_obj_path   = "data/objects/clipper/clipper.obj",
    initial_grasp_path = "data/objects/wrench/wrench_engage.yaml",
    if_FPSA = False,
    fpsa_aug_root = "./data/objects/bracket/fpsa_aug_outputs",
    fpsa_include_base = True,
    obj_pose_base = [0.75, -0.05, 0.25],
    obj_euler_base = [0.0, 0.0, 0.0],# screw is a hexagon, 0-60 covers all space
    randomize_lighting= True,
    # outlier scene         
    randomize_outlscene  = True,
    outlscene_xyz_jit    = 0.015,
    outlscene_eul_jit    = 0.001,
    # plane height randomization
    randomize_plane_height = True,
    plane_height_jit = 0.002,
    randomize_objpose  = True,
    obj_x_jit    = 0.06,
    obj_y_jit    = 0.1,
    obj_z_jit    = 0.06,
    obj_z_eul_jit = np.pi / 6,
    randomize_campose = True,
    cam_xyz_jit  = 0.01,
    cam_eul_jit  = 0.005,
    randomize_image_noise= True,
    randomize_object_color = True,
    object_color_mode = "bounded",  # "bounded" or "recolor"
    object_color_strength = 0.5,
    wrench_color_mode= "bounded",
    wrench_color_strength= 0.1,
    randomize_distractors= True,
    distractor_root= "/mnt/storage/GoogleScannedObjects",
    distractor_num_range= (0, 5),
    distractor_target_size_range= (0.06, 0.4),
    distractor_workspace = ((-0.2, 0.8), (-0.72, 0.42)),
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

        if sim_step % record_every_n_sim_steps == 0:
            _, _, RGB, _, _ = Sim.get_agentview_image()
            cv2.imwrite("temp_rgb.png", cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR))
            obs = Sim.collect_observation()

            record_idx += 1
            # print(obs["robot0_gripper_qpos"])
            print(record_idx)
        # time.sleep(0.05)
            print(Sim.is_success())

        #Sim.collect_action()

except KeyboardInterrupt:
    print("Stopped by user")
