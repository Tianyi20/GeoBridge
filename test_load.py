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


def load_models(bullet_client,
                visual_mesh_file, 
                vhacd_mesh_file, 
                desired_mass = 1.0,
                position = [0.0,0.0,0.0],
                baseOrientation = [0,0,0,1],
                center_of_mass = [0.0,0.0,0.0],
                lateral_friction = 0.6,
                spinning_friction = 0.003,
                visual_only=False,
                ):
    """
    Load a body into the simulation, given the visual mesh file and the vhacd mesh file.
    Position should be zero if the mesh's origin is at world origin.
    Center of mass very important, should be calculated before loading.
    Returns the body id of the loaded body.
    """
    visual_shape_id = bullet_client.createVisualShape(shapeType=bullet_client.GEOM_MESH,
                                          fileName=visual_mesh_file,
                                          meshScale=[1, 1, 1])
    # Optional collision shape
    if visual_only:
        collision_shape_id = -1
    else:
        assert vhacd_mesh_file is not None, \
            "vhacd_mesh_file is required unless visual_only=True"
        collision_shape_id = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=vhacd_mesh_file,
            meshScale=[1, 1, 1],
        )
    
    body_id = bullet_client.createMultiBody(baseMass= desired_mass,  # 你可以根据需要指定质量
                                baseCollisionShapeIndex=collision_shape_id,
                                baseVisualShapeIndex=visual_shape_id,
                                basePosition=position,
                                # 这里的 baseInertialFramePosition 是 center of mass 相对于 baseposition. very importatnt property. 的pose。-0.01743, -0.08127,  0.02427
                                baseInertialFramePosition=center_of_mass, 
                                #baseInertialFrameOrientation=[0, 0, 0, 1],
                                baseOrientation=baseOrientation)
    if not visual_only:
        bullet_client.changeDynamics(body_id, -1, lateralFriction = lateral_friction, 
                            spinningFriction = spinning_friction,
                            restitution = 0.5,
                            rollingFriction = 0.005)

    return body_id

import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")


load_models(p, visual_mesh_file= "/home/iadc/GeoBridge/data/objects/bracket/bracket_texture/bracket.obj",
            vhacd_mesh_file= "./data/objects/bracket/bracket_coacd.obj")

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
