import random

import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
#from CupOnRackSim import CupOnRackSim, CupOnRackSimAuto
from PickUpSim import CupOnRackSim
from episode_writer import EpisodeWriter
import json
from pathlib import Path
from icecream import ic
    
def collect_one_episode(sim :CupOnRackSim, episode_dir, fps=20, 
                        max_steps=3000, record_every_n_sim_steps=12,
                        cluster_offset=None,  
                        z_rot=None ):
    """
    PyBullet control_dt = 1/240
    如果每12个sim step录一帧 -> 20 FPS
    """
    writer = EpisodeWriter(episode_dir, 
                           fps=fps,
                           extra_meta={
                               "cluster_offset": cluster_offset,
                               "z_rot": z_rot,
                               }
                            )

    sim.done = False
    sim_step = 0
    timestamp = 0.0
    record_idx = 0

    while (not sim.done) and sim_step < max_steps:
        if sim_step % record_every_n_sim_steps == 0:
            obs = sim.collect_observation()   # 先采当前观测 s_t
        # 计算并下发本步控制
        sim.step()   

        if sim_step % record_every_n_sim_steps == 0:
            action = sim.collect_action()
            # ic(sim_step, timestamp, record_idx, 
            #    obs['robot0_eef_pos'], obs['robot0_eef_quat'], 
            #    target_pos, target_orn)
            timestamp = record_idx / float(fps)
            writer.add_step(obs, action, timestamp)
            record_idx += 1

        p.stepSimulation()
        sim_step += 1
    success = sim.is_success()
    ic("if success", success)
    writer.close(success=success)



p.connect(p.DIRECT)
# turn off GUI shadows 
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.setAdditionalSearchPath(pd.getDataPath())
timeStep=1./240.
p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

# seed
np.random.seed(42)


NUM_EPISODES = 300
BASE_DIR = Path("./collected_episodes")

for ep in range(NUM_EPISODES):
    print(f"\n=== Collecting episode {ep:06d} ===")

    p.setTimeStep(1. / 240.)
    p.setGravity(0, 0, -9.8)
    episode_dir = BASE_DIR / f"episode_{ep:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    z_rot = math.pi /2
    cluster_offset = [random.uniform(0.3, 0.5), random.uniform(0.3, 0.4), 0.0]
    panda = CupOnRackSim(
        p,
        offset=[0, 0, 0],
        cluster_offset= cluster_offset,
        z_axis_rotation= z_rot
    )

    collect_one_episode(
        sim=panda,
        episode_dir=episode_dir,
        cluster_offset=cluster_offset,
        z_rot=z_rot,
    )

    p.resetSimulation()