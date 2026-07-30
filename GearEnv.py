import math
import os
import json
import gym
from gym import spaces
from icecream import ic
import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet as p
from GearSim import GearSim


class GearEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, sim_steps_per_action=1,
                 connection_mode = p.DIRECT,
                 seed=46,
                 use_agent_cam = True,
                 use_fisheye_cam = True,
                 if_FPSA = False,
                 randomize_image_noise  = True,
                 randomize_objcolor = True,
                 randomize_lighting = True,
                 randomize_objpose  = True,
                 randomize_distractors  = True,
                 randomize_outlscene = True,
                 randomize_plane_height = True,
                 randomize_campose = True,
                 randomize_wrenchpose = True,
                 randomize_wrenchcolor = True,
        ):
        self._seed = seed
        self.use_agent_cam = use_agent_cam
        self.use_fisheye_cam = use_fisheye_cam
        self.if_FPSA = if_FPSA
        self.randomize_image_noise = randomize_image_noise
        self.randomize_objcolor = randomize_objcolor
        self.randomize_lighting = randomize_lighting
        self.randomize_objpose = randomize_objpose
        self.randomize_distractors = randomize_distractors
        self.randomize_outlscene = randomize_outlscene
        self.randomize_plane_height = randomize_plane_height
        self.randomize_wrenchpose = randomize_wrenchpose
        self.randomize_wrenchcolor = randomize_wrenchcolor
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
            # "agentview_image": spaces.Box(
            #     low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            # ),
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

        if self.use_agent_cam:
            self.observation_space.spaces["agentview_image"] = spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            )
        if self.use_fisheye_cam:
            self.observation_space.spaces["robot0_eye_in_hand_image"] = spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            )    


    def seed(self, seed):
        self._seed = seed

    def _build_sim(self):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setTimeStep(1. / 120.)
        self._pybullet_client.setGravity(0, 0, -9.8)

        self.sim = GearSim(
            bullet_client=self._pybullet_client,
            control_dt = 1. / 120.,
            offset=[0, 0, 0],
            seed=self._seed,
            randomize_initial_ee_pose= True
        )

        self.sim.enable_high_quality_rendering()

        self.sim.make_scene(
            env_mesh_path="./data/background/repaired_table/tabletop.obj",
            gear_path="./data/objects/gear_extraction/gear/gear.obj",
            supportor_path="./data/objects/gear_extraction/supportor/supportor.obj",
            initial_grasp_path="./data/objects/gear_extraction/gear/grasp.yaml",
            gear_collision_path=None,
            supportor_collision_path=None,

            if_FPSA_gear=True,
            fpsa_gear_aug_root="./data/objects/gear_extraction/gear/Gear_aug_outputs",
            fpsa_gear_include_base=True,

            # The gear origin is located at supportor-local (0, 0, 0.31).
            supportor_pose_base=[0.62, 0.10, 0.015],
            supportor_euler_base=[0.0, 0.0, 0.0],
            gear_to_support_pos=[0.0, 0.0, 0.32],
            gear_to_support_euler=[0.0, np.pi, np.pi],

            # Nearby low placement target, also expressed in the supportor frame.
            support_to_place_pos=[-0.10, 0.0, 0.1],
            support_to_place_euler=None,  # preserve the extracted gear orientation
            randomize_task_pose=True,
            task_x_jit=0.10,
            task_y_jit=0.06,
            task_z_jit=0.03,
            task_yaw_jit= 0.0,

            safe_approach=0.03,
            lift_height=0.03,
            place_approach_height=0.04,
            retreat_height=0.0,

            # Same visual-domain randomization configuration as AssemblyDemo.
            randomize_lighting=True,
            randomize_outlscene=True,
            outlscene_xyz_jit=0.015,
            outlscene_eul_jit=0.001,
            randomize_plane_height=True,
            plane_height_jit=0.003,
            randomize_campose=True,
            cam_xyz_jit=0.01,
            cam_eul_jit=0.005,
            randomize_fisheye_cam=True,
            fisheye_eyz_jit=0.005,
            fisheye_eul_jit=0.002,
            randomize_camera_intrinsic=True,
            randomize_image_noise=True,
            randomize_object_color=True,
            randomize_robot_texture=True,
            robot_texture_patterns=("checkers", "gradient", "noise", "plain"),
            object_color_mode="bounded",
            object_color_strength=0.5,
            randomize_distractors=True,
            distractor_root="/mnt/storage/GoogleScannedObjects",
            distractor_num_range=(0, 5),
            distractor_target_size_range=(0.1, 0.5),
            distractor_workspace=((-0.2, 1.3), (-0.72, 0.42)),
            distractor_clearance=0.06,
            distractor_path_clearance=0.06,
            distractor_min_target_mask_pixels=10,

            # Keep AssemblySim's post-grasp pose randomization, but use a conservative
            # range so the extracted gear does not get teleported into the support pin.
            fix_gear_to_gripper=True,
            randomize_object_in_hand_pose=True,
            object_in_hand_x_jit=0.002,
            object_in_hand_y_jit=0.002,
            object_in_hand_z_jit=0.002,
            object_in_hand_roll_jit= 0.002,
            object_in_hand_pitch_jit= 0.002,
            object_in_hand_yaw_jit= 0.002,
            object_in_hand_debug=False,
        )

    def reset(self):
        self._build_sim()
        obs = self.sim.collect_observation(use_agent_cam= self.use_agent_cam, 
                                           direct= True, 
                                           use_eye_in_hand= self.use_fisheye_cam,
                                           collect_gear_ee= False)
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
        # always use parent tcp as frame for real deploy
        self.sim.solve_ik_and_apply(target_pos, target_orn, input_frame= "parent_tcp")
        # Set gripper status
        self.sim.set_gripper(gripper)

        for _ in range(self.sim_steps_per_action):
            self._pybullet_client.stepSimulation()

        # use native get obs and is success
        obs = self.sim.collect_observation(use_agent_cam= self.use_agent_cam, 
                                           direct= True, 
                                           use_eye_in_hand= self.use_fisheye_cam,
                                           collect_gear_ee= False)
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

    def render(self, mode="rgb_array"):
        obs = self._last_obs
        imgs = []

        if self.use_agent_cam:
            imgs.append(obs["agentview_image"])

        if self.use_fisheye_cam:
            imgs.append(obs["robot0_eye_in_hand_image"])

        if len(imgs) == 0:
            return None

        if len(imgs) == 1:
            return imgs[0]

        gap = 16
        h = imgs[0].shape[0]
        spacer = np.full((h, gap, 3), 255, dtype=np.uint8)

        return np.concatenate([imgs[0], spacer, imgs[1]], axis=1)

    def close(self):
        if self._pybullet_client is not None:
            self._pybullet_client.disconnect()
            self._pybullet_client = None