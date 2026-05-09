import math
import os
import random
import gym
from gym import spaces
from icecream import ic
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation
from PickUpSim import CupOnRackSim
import json


class CupOnRackEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, sim_steps_per_action=1,
                 connection_mode = p.GUI):
        
        super(CupOnRackEnv, self).__init__()
        self.sim_steps_per_action = sim_steps_per_action
        # PyBullet: GUI 就不能vectorize,
        self._pybullet_client = bc.BulletClient(connection_mode=connection_mode)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        self.sim = None

        # Rendering ultility
        self._last_obs = None

        # action = [target_pos(3), target_orn(4), gripper(1)]
        self.action_space = spaces.Box(
            low=np.array([-np.inf] * 8, dtype=np.float32),
            high=np.array([np.inf] * 8, dtype=np.float32),
            dtype=np.float32,
        )

        # collect_observation() 的输出格式
        self.observation_space = spaces.Dict({
            "agentview_image": spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            ),
            "robot0_eye_in_hand_image": spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            ),
            "robot0_eef_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "robot0_eef_quat": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            "robot0_gripper_qpos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
        })


    def _build_with_meta(self, meta_path = None):
        if not os.path.exists(meta_path):
            print(f"Fail: missing episode_meta.json")
            exit()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._build_sim(meta["cluster_offset"], meta["z_rot"])

    def seed(self, seed):
        np.random.seed(seed)

    def _build_sim(self, cluster_offset = None, z_rot = None):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setTimeStep(1. / 240.)
        self._pybullet_client.setGravity(0, 0, -9.8)
        
        if z_rot is None:
            z_rot = math.pi /2
        if cluster_offset is None:
            cluster_offset = [random.uniform(0.3, 0.5), random.uniform(0.3, 0.4), 0.0]

        # ic(cluster_offset)
        self.sim = CupOnRackSim(
            bullet_client=self._pybullet_client,
            offset=[0, 0, 0],
            cluster_offset= cluster_offset,
            z_axis_rotation= z_rot,
        )

    def reset(self):
        self._build_sim()
        obs = self.sim.collect_observation()
        self._last_obs = obs
        return obs

    def step(self, action):
        """
        action shape: (8,)
        relative pose, axis angle, gripper 
        action = [x, y, z, qx, qy, qz, qw, gripper]
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # ic(action.shape)
        assert action.shape[0] == 8, "action must be shape (8,)"

        # parse action
        target_pos = action[:3]
        target_orn = action[3:7]
        gripper = float(action[7])
        gripper = np.clip(gripper, 0.0, 0.04)

        # use sim native IK and step
        self.sim.solve_ik_and_apply(target_pos, target_orn)
        self.sim.set_gripper(gripper)

        for _ in range(self.sim_steps_per_action):
            self._pybullet_client.stepSimulation()

        # use native get obs and is success
        obs = self.sim.collect_observation()
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