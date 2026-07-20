import math
import os
import json
import gym
from gym import spaces
from icecream import ic
import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet as p
from AssemblySim import AssemblySim


class AssemblyEnv(gym.Env):
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

        self.sim = AssemblySim(
            bullet_client=self._pybullet_client,
            control_dt = 1. / 120.,
            offset=[0, 0, 0],
            seed=self._seed,
            randomize_initial_ee_pose= True
        )

        self.sim.enable_high_quality_rendering()

        self.sim.make_scene(
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
            randomize_distractors= True,
            distractor_root= "/mnt/storage/GoogleScannedObjects",
            distractor_num_range= (0, 5),
            distractor_target_size_range= (0.1, 0.5),
            distractor_workspace = ((-0.2, 1.3), (-0.72, 0.42)),
            distractor_clearance = 0.06,
            distractor_path_clearance = 0.06,
            # at least 10 pixel of the target object
            distractor_min_target_mask_pixels= 10,
            fix_parent_to_gripper=True,
            randomize_object_in_hand_pose=True,
            object_in_hand_x_jit=0.025,
            object_in_hand_y_jit=0.025,
            object_in_hand_z_jit=0.02,
            object_in_hand_roll_jit=0.0174533,
            object_in_hand_pitch_jit=0.0174533,
            object_in_hand_yaw_jit=0.0349066,
            object_in_hand_debug=False,
        )

    def reset(self):
        self._build_sim()
        obs = self.sim.collect_observation(use_agent_cam= self.use_agent_cam, 
                                           direct= True, 
                                           use_eye_in_hand= self.use_fisheye_cam,
                                           collect_wrench_ee= False)
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
                                           collect_wrench_ee= False)
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