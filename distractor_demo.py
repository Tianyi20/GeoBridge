from PickUpSim import PickUpSim
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
p.connect(p.DIRECT)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.setAdditionalSearchPath(pd.getDataPath())
timeStep=1./240.
p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

Sim = PickUpSim(p, offset=[0, 0, 0], seed = 4122623382)

Sim.make_scene(env_mesh_path= "./data/background/patched_table/tabletop.obj",
               manipulated_obj_path= "./data/objects/banana/banana.obj",
               initial_grasp_path= "./data/objects/banana/grasp.yaml",
               obj_pose_base = [0.5, 0.0, 0.1],
               obj_euler_base = [math.pi/2, 0.0, math.pi/2],
               randomize_image_noise= True,
               randomize_lighting= True,
               randomize_objpose= True,
               randomize_distractors= True,
               distractor_root= "/mnt/storage/GoogleScannedObjects",
               distractor_num_range= (0, 4),
               distractor_target_size_range= (0.06, 0.3),
               distractor_workspace = ((0.05, 0.78), (-0.42, 0.42)),
               # at least 10 pixel of the target object
               distractor_min_target_mask_pixels= 10,)
Sim.enable_high_quality_rendering()

sim_step = 0
timestamp = 0.0
record_idx = 0
record_every_n_sim_steps = 24
try:
    while (not Sim.done):
        p.stepSimulation()
        Sim.step()
        sim_step += 1

        if sim_step % record_every_n_sim_steps == 0:
            _, _, RGB, _, _ = Sim.get_agentview_image()
            cv2.imwrite("temp_rgb.png", cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR))
        # time.sleep(0.05)
        # print(Sim.is_success())

except KeyboardInterrupt:
    print("Stopped by user")
