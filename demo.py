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
p.connect(p.GUI)
# turn off GUI shadows 
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.setAdditionalSearchPath(pd.getDataPath())
timeStep=1./240.
p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

Sim = PickUpSim(p, offset=[0, 0, 0])
Sim.make_scene(env_mesh_path= "./data/background/patched_table/tabletop.obj",
               manipulated_obj_path= "./data/objects/banana/banana.obj",
               initial_grasp_path= None,
               obj_pose_offset = [0.5, 0.0, 0.1],
               obj_euler_offset = [math.pi/2, 0.0, math.pi/2],)

rgb = Sim.get_agentview_image()
cv2.imwrite("camera.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
try:
    while True:
        p.stepSimulation()
except KeyboardInterrupt:
    print("Stopped by user")
