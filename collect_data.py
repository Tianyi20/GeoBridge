import random
import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
from PickUpSim import PickUpSim
from episode_writer import EpisodeWriter
import json
from pathlib import Path
from icecream import ic
from tqdm import tqdm
from datetime import datetime

def collect_one_episode(sim :PickUpSim, 
                        episode_dir,
                        meta_seed,
                        fps=10, 
                        max_steps=2000, 
                        record_every_n_sim_steps=12,
                        ):
    """
    PyBullet control_dt = 1/120
    如果每12个sim step录一帧 -> 10 FPS
    """

    ## init episode writer
    writer = EpisodeWriter(episode_dir, 
                           fps=fps,
                           if_agent_view  = True,
                           if_eye_in_hand = False, 
                           extra_meta={"meta_seed": meta_seed,}
                         )

    sim.done = False
    sim_step = 0
    timestamp = 0.0
    record_idx = 0

    while (not sim.done) and sim_step < max_steps:
        if sim_step % record_every_n_sim_steps == 0:
            obs = sim.collect_observation(direct= False)   # 先采当前观测 s_t
        # 计算并下发本步控制
        sim.step()   

        if sim_step % record_every_n_sim_steps == 0:
            action = sim.collect_action()
            # ic(sim_step, timestamp, record_idx, 
            #    obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']
            #    )
            timestamp = record_idx / float(fps)
            writer.add_step(obs, action, timestamp)
            record_idx += 1

        p.stepSimulation()
        sim_step += 1
    success = sim.is_success()
    ic("if success", success)
    writer.close(success=success)


if __name__ == "__main__":
    p.connect(p.DIRECT)
    # turn off GUI shadows 
    # seed
    np.random.seed(42)
    # seed env randomization
    _seed = 43
    NUM_EPISODES = 5
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = Path("/mnt/storage/DP_data/pickup/")/ run_time / "episodes"
    
    ic(BASE_DIR)
    for ep in tqdm(range(NUM_EPISODES)):
        print(f"\n=== Collecting episode {ep:06d} ===")
        _seed += 1
        print(f"seed: {_seed}")

        timeStep=1./120.
        p.setTimeStep(timeStep)
        p.setGravity(0, 0, -9.8)
        episode_dir = BASE_DIR / f"episode_{ep:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        Sim = PickUpSim(p, offset=[0, 0, 0], control_dt = timeStep, seed = _seed)

        Sim.make_scene(
            env_mesh_path= "./data/background/repaired_table/tabletop.obj",
            manipulated_obj_path= "./data/objects/bracket/bracket_texture/bracket.obj",
            initial_grasp_path= "./data/objects/bracket/bracket_texture/bracket_grasp.yaml",
            if_FPSA = True,
            fpsa_aug_root = "./data/objects/bracket/fpsa_aug_outputs",
            fpsa_include_base = True,
            obj_pose_base = [0.55, 0.0, 0.1],
            obj_euler_base = [0.0, 0.0, 0.0],
            randomize_lighting= True,
            # outlier scene         
            randomize_outlscene  = True,
            outlscene_xyz_jit    = 0.015,
            outlscene_eul_jit    = 0.001,
            # plane height randomization
            randomize_plane_height = True,
            plane_height_jit = 0.002,
            randomize_objpose  = True,
            obj_x_jit    = 0.15,
            obj_y_jit    = 0.2,
            obj_z_eul_jit = np.pi,
            randomize_campose = True,
            cam_xyz_jit  = 0.004,
            cam_eul_jit  = 0.002,
            randomize_image_noise= True,
            randomize_object_color = True,
            object_color_mode = "bounded",  # "bounded" or "recolor"
            object_color_strength = 1.0,
            randomize_distractors= True,
            distractor_root= "/mnt/storage/GoogleScannedObjects",
            distractor_num_range= (0, 5),
            distractor_target_size_range= (0.06, 0.4),
            distractor_workspace = ((-0.2, 0.8), (-0.72, 0.42)),
            # at least 10 pixel of the target object
            distractor_min_target_mask_pixels= 10,
            )
        Sim.enable_high_quality_rendering()

        collect_one_episode(
            sim = Sim,
            episode_dir = episode_dir,
            meta_seed = _seed,
        )

        p.resetSimulation()