import math
import os
import json
import gym
from gym import spaces
from icecream import ic
import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet as p
from PickUpSim import PickUpSim


class PickUpEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, sim_steps_per_action=1,
                 connection_mode = p.DIRECT,
                 seed=42,
                 if_FPSA = True,
                 randomize_image_noise  = True,
                 randomize_objcolor = True,
                 randomize_lighting = True,
                 randomize_objpose  = True,
                 randomize_distractors  = True,
                 randomize_outlscene = True,
                 randomize_plane_height = True,
                 randomize_campose = True,
        ):
        self._seed = seed
        self.if_FPSA = if_FPSA
        self.randomize_image_noise = randomize_image_noise
        self.randomize_objcolor = randomize_objcolor
        self.randomize_lighting = randomize_lighting
        self.randomize_objpose = randomize_objpose
        self.randomize_distractors = randomize_distractors
        self.randomize_outlscene = randomize_outlscene
        self.randomize_plane_height = randomize_plane_height
        self.randomize_campose = randomize_campose
        self.sim_steps_per_action = sim_steps_per_action
        
        # PyBullet: GUI 就不能vectorize,
        self._pybullet_client = bc.BulletClient(connection_mode=connection_mode)
        self.sim = None

        # Rendering ultility
        self._last_obs = None

        # gym dependent action space and observation space
        self.action_space = spaces.Box(
            low=np.array([-np.inf] * 8, dtype=np.float32),
            high=np.array([np.inf] * 8, dtype=np.float32),
            dtype=np.float32,
        )

        # same as collect_observation()
        self.observation_space = spaces.Dict({
            "agentview_image": spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            ),
            # "robot0_eye_in_hand_image": spaces.Box(
            #     low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            # ),
            "robot0_eef_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "robot0_eef_quat": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            "robot0_gripper_qpos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
        })


    def seed(self, seed):
        self._seed = seed

    def _build_sim(self):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setTimeStep(1. / 120.)
        self._pybullet_client.setGravity(0, 0, -9.8)

        self.sim = PickUpSim(
            bullet_client=self._pybullet_client,
            control_dt = 1. / 120.,
            offset=[0, 0, 0],
            seed=self._seed,
        )

        self.sim.enable_high_quality_rendering()

        self.sim.make_scene(
                env_mesh_path= "./data/background/repaired_table/tabletop.obj",
                manipulated_obj_path= "./data/objects/bracket/bracket_texture/bracket.obj",
                initial_grasp_path= "./data/objects/bracket/bracket_texture/bracket_grasp.yaml",
                if_FPSA= self.if_FPSA,
                fpsa_aug_root = "./data/objects/bracket/fpsa_aug_outputs",
                fpsa_include_base = True,                
                obj_pose_base = [0.55, 0.0, 0.1],
                obj_euler_base = [0.0, 0.0, 0.0],
                randomize_lighting = self.randomize_image_noise,
                # outlier scene         
                randomize_outlscene  = self.randomize_outlscene,
                outlscene_xyz_jit    = 0.015,
                outlscene_eul_jit    = 0.002,
                # plane height randomization
                randomize_plane_height = self.randomize_plane_height,
                plane_height_jit = 0.002,
                randomize_objpose  = self.randomize_objpose,
                obj_x_jit    = 0.15,
                obj_y_jit    = 0.2,
                obj_z_eul_jit = np.pi,
                randomize_campose =  self.randomize_campose,
                cam_xyz_jit  = 0.004,
                cam_eul_jit  = 0.002,
                randomize_image_noise= self.randomize_image_noise,
                randomize_object_color = self.randomize_objcolor,
                object_color_mode = "bounded",  # "bounded" or "recolor"
                object_color_strength = 1.0,
                randomize_distractors= self.randomize_distractors,
                distractor_root= "/mnt/storage/GoogleScannedObjects",
                distractor_num_range= (0, 5),
                distractor_target_size_range= (0.06, 0.4),
                distractor_workspace = ((-0.2, 0.8), (-0.72, 0.42)),
                # at least 10 pixel of the target object
                distractor_min_target_mask_pixels= 10,
                )

    def reset(self):
        self._build_sim()
        obs = self.sim.collect_observation()
        self._last_obs = obs
        return obs

    def step(self, action):
        """
        action shape: (8,)
        pose, quaternion, gripper 
        action = [x, y, z, qx, qy, qz, qw, gripper]
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # ic(action.shape)
        assert action.shape[0] == 8, "action must be shape (8,)"

        # parse action
        target_pos = action[:3]
        target_orn = action[3:7]
        gripper = (action[7])

        # use sim native IK and step
        self.sim.solve_ik_and_apply(target_pos, target_orn)
        self.sim.set_gripper(gripper)

        for _ in range(self.sim_steps_per_action):
            self._pybullet_client.stepSimulation()

        # use native get obs and is success
        obs = self.sim.collect_observation(direct= False)
        self._last_obs = obs
        done = self.sim.is_success()
        # reward is done, sparse reward
        reward = done

        info = {
            "target_pos": target_pos,
            "target_orn": target_orn,
            "gripper": gripper,
        }

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        return self._last_obs["agentview_image"]

    def close(self):
        if self._pybullet_client is not None:
            self._pybullet_client.disconnect()
            self._pybullet_client = None