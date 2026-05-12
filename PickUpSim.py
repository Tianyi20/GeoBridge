import time
import cv2
import gym
import numpy as np
import math
import pybullet_data
from utility import load_initial_grasp_pose
from VisualDR.lightingDR import LightingDR
from VisualDR.ImgNoiseDR import ImgNoiseDR
from VisualDR.objposeDR import ObjPoseDR
from VisualDR.distractorDR import DistractorDR

from pybullet_utility import (
    load_models,
    coacd_convex_decomposition,
    get_com,
    get_true_PositionAndOrientation,
    get_COM_PositionAndOrientation,
    quat_diff_rotvec,
    quat_slerp,
    cvK2BulletP,
    cvPose2BulletView
)

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
ul = [7] * pandaNumDofs
jr = [7] * pandaNumDofs

jointPositions = [
    -0.0768761337796847,
    0.1692503838434554,
    -0.5782208367480097,
    -1.4272947420449444,
    0.055714113760170644,
    1.5859262946844102,
    -1.0857437887214538,
    0.02,
    0.02,
]
rp = jointPositions


class PickUpSim(object):
    def __init__(self, bullet_client, offset, seed = 42):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

        # Visual domain randomizers. Each randomizer owns its own RNG and state.
        # Keep bullet_client inside every module so each multiprocessing worker can
        # create an independent PickUpSim / VisualDR stack.
        self.lightingDR = LightingDR(self.bullet_client, seed=seed)
        self.ImgNoiseDR = ImgNoiseDR(self.bullet_client, seed=seed)
        self.objposeDR = ObjPoseDR(self.bullet_client, seed=seed)
        self.distractorDR = DistractorDR(self.bullet_client, seed=seed)

        self.offset = np.array(offset)
        self.control_dt = 1.0 / 240.0
        self.t = 0.0
        # Set GUI's Sky to white, only influence GUI
        self.bullet_client.configureDebugVisualizer(rgbBackground=[1, 1, 1])
        self.finger_target = 0.04
        self.safe_approach = 0.1
        self.safe_grasp_offset = 0.05
        self.arm_force = 200.0
        self.gripper_force = 100.0

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        base_orn = self.bullet_client.getQuaternionFromEuler([0, 0, 0])

        ################## Load Panda robot #################
        self.panda = self.bullet_client.loadURDF(
            "franka_panda/panda.urdf",
            np.array([0, 0, 0]) + self.offset,
            base_orn,
            useFixedBase=True,
            flags=flags,
        )
        c = self.bullet_client.createConstraint(
            self.panda, 9, self.panda, 10,
            jointType=self.bullet_client.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=100)

        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointType = info[2]

            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index += 1
            if jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index += 1
        # Home pose and joints
        self.home_joint = np.array(rp[:7], dtype=float)
        self.home_ee_pos, self.home_ee_orn = self.get_ee_pose()
        self.home_ee_pos = np.array(self.home_ee_pos, dtype=float)
        self.home_ee_orn = np.array(self.home_ee_orn, dtype=float)

        ########333#========= pick up状态机 =========###################
        self.states = [
            "home",
            "move_pregrasp",
            "open_gripper",
            "move_grasp",
            "close_gripper",
            "lift_object",
        ]
        self.state_durations = [0.01, 1.0, 0.15, 0.8, 0.3, 1.0]
        self.state_idx = 0
        self.state = self.states[self.state_idx]
        self.state_t = 0.0

        # target pose, target gripper
        self.target_pos = None
        self.target_orn = None
        self.target_gripper = None
        self.done = False

        # 当前 state 对应的一段轨迹
        self.motion_start_pos = None
        self.motion_start_orn = None
        self.motion_target_pos = None
        self.motion_target_orn = None

        # 抓取参考
        self.last_grasp_pose = None
        self.last_grasp_orn = None

        self.prepare_state(self.state)

    def make_scene(self, 
                   env_mesh_path        = None,
                   manipulated_obj_path = None,
                   initial_grasp_path = None,
                   obj_pose_base = [0.5, 0.0, 0.0],
                   obj_euler_base = [0.0, 0.0, 0.0],
                   randomize_lighting = True,
                   randomize_objpose  = True,
                   randomize_image_noise = True,
                   randomize_distractors = True,
                   distractor_root = "/mnt/storage/GoogleScannedObjects",
                   distractor_num_range = (1, 5),
                   distractor_target_size_range = (0.06, 0.16),
                   distractor_workspace = ((0.25, 0.78), (-0.42, 0.42)),
                   distractor_clearance = 0.035,
                   distractor_path_clearance = 0.10,
                   distractor_min_target_mask_pixels = 1,
                   ):
        
        ## TODO: visual things to be randomized:
        # 1. random number and shape of distractors (no need accurate convex decomposition)
        # 2. slight disturbance of the object texture, ground texture
        # 3. objects pose and z axis orientation 
        # 4. number of lights in the scene
        # 5. position, orientation, specular characteristics of the lights
        # 6. types and amount of random noise added to the images 

        # always enable high quality rendering pipeline and shadows

        """ ################ Load basic scene assets ################ """
        # Avoid mutating caller-provided lists/default arguments when pose randomization is enabled.
        obj_pose_base = np.array(obj_pose_base, dtype=float).copy()
        obj_euler_base = np.array(obj_euler_base, dtype=float).copy()

        self.env_mesh_path = env_mesh_path
        self.pick_up_obj_path = manipulated_obj_path
        self.convex_pick_up_obj_path = coacd_convex_decomposition(self.pick_up_obj_path)
        self.com_pick_up_obj = get_com(self.pick_up_obj_path)

        """ ################ Basic Visual Randomization ############### """
        if randomize_lighting:
            self.lightingDR.sample_lighting_randomization()
        else:
            self.lightingDR.reset_to_default()
        
        if randomize_objpose:
            obj_xy_jitter, obj_z_axis_rotation_jitter = self.objposeDR.sample_obj_pose_randomization(
                xy_jitter_range=0.2,
                z_axis_rotation_range=np.pi,
            )
            obj_pose_base[:2] += obj_xy_jitter
            obj_euler_base[2] += obj_z_axis_rotation_jitter

        if randomize_image_noise:
            self.ImgNoiseDR.sample_image_noise_randomization()
        else:
            self.ImgNoiseDR.reset()

        # Load ground plane but make it invisible
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load PyBullet's default ground plane
        self.ground_plane_id = self.bullet_client.loadURDF(
            "plane.urdf",
            basePosition=[0, 0, 0],
        )

        # Make the plane invisible but keep collision
        self.bullet_client.changeVisualShape(
            self.ground_plane_id,
            -1,
            rgbaColor=[1, 1, 1, 0],  # alpha = 0
        )
        # load background, but make it non-collidable
        self.env_mesh = load_models(
            self.bullet_client,
            visual_mesh_file=self.env_mesh_path,
            vhacd_mesh_file=None,
            desired_mass=0.0,
            visual_only=True,
        )
        
        self.pick_up_obj_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.pick_up_obj_path,
            vhacd_mesh_file=self.convex_pick_up_obj_path,
            desired_mass=0.5,
            position=np.array(obj_pose_base),
            baseOrientation=self.bullet_client.getQuaternionFromEuler(np.array(obj_euler_base)),
            center_of_mass=np.array(self.com_pick_up_obj),
            lateral_friction=0.6,
        )
        self.initial_grasp_guess = load_initial_grasp_pose(initial_grasp_path)

        # Let the manipulated object settle first. The resulting AABB/grasp pose is
        # used to reject distractor placements around the target and robot path.
        self.waite_scene_stable()


        """############# Distractor Domain randomization ##############"""
        if randomize_distractors:
            self.distractorDR.sample_and_load_distractors(
                distractor_root=distractor_root,
                num_range=distractor_num_range,
                target_size_range=distractor_target_size_range,
                workspace=distractor_workspace,
                clearance=distractor_clearance,
                path_clearance=distractor_path_clearance,
                min_target_mask_pixels=distractor_min_target_mask_pixels,
                target_body_id=self.pick_up_obj_id,
                robot_body_id=self.panda,
                robot_base_offset=self.offset,
                planned_waypoints=self.get_state_machine_ee_waypoints(),
                render_agentview_fn=self.get_agentview_image,
                end_effector_index=pandaEndEffectorIndex,
                ik_lower_limits=ll,
                ik_upper_limits=ul,
                ik_joint_ranges=jr,
                get_current_arm_joints_fn=self.get_current_arm_joints,
                quat_slerp_fn=quat_slerp,
                panda_num_dofs=pandaNumDofs,
            )
        else:
            self.distractorDR.clear_distractors()

        # Step a few frames after static distractor creation so broad-phase contacts
        # are updated before collecting observations.
        for _ in range(20):
            self.bullet_client.stepSimulation()

        obj_pos, obj_orn = get_true_PositionAndOrientation(
            self.bullet_client, self.pick_up_obj_id
        )
        self.initial_obj_pos = np.array(obj_pos, dtype=float)
        self.initial_obj_orn = np.array(obj_orn, dtype=float)

    def get_state_machine_ee_waypoints(self):
        """Approximate the EE path used by the pick-up state machine."""
        grasp_pos, grasp_orn = self.get_initial_guess_grasp()

        pregrasp_pos = grasp_pos.copy()
        pregrasp_pos[2] += self.safe_approach

        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.2

        return [
            (self.home_ee_pos.copy(), self.home_ee_orn.copy()),
            (pregrasp_pos, grasp_orn.copy()),
            (grasp_pos.copy(), grasp_orn.copy()),
            (lift_pos, grasp_orn.copy()),
        ]

    def waite_scene_stable(self, waite_steps=1000, vel_threshold=0.005):       
        steps = 0 
        while steps < waite_steps:
            self.bullet_client.stepSimulation()
            steps += 1
            vel1, ang_vel1 = self.bullet_client.getBaseVelocity(self.pick_up_obj_id)

            speed1 = np.linalg.norm(vel1) + np.linalg.norm(ang_vel1)

            if speed1 < vel_threshold:
                print("Scene stabilized.")
                return True

        print("Warning: Scene did not stabilize within timeout.")
        return False

    def get_initial_guess_grasp(self):
        mesh_world_pos, mesh_world_orn = get_true_PositionAndOrientation(self.bullet_client,
                                                                         self.pick_up_obj_id)
        grasp_pose, grasp_orn = self.bullet_client.multiplyTransforms(
            mesh_world_pos,
            mesh_world_orn,
            self.initial_grasp_guess["t"], # initial grasp is defined manually
            self.initial_grasp_guess["quat"],
        )
        return np.array(grasp_pose, dtype=float), np.array(grasp_orn, dtype=float)
    
    def get_ee_pose(self):
        link_state = self.bullet_client.getLinkState(
            self.panda, pandaEndEffectorIndex, computeForwardKinematics=True
        )
        return np.array(link_state[4], dtype=float), np.array(link_state[5], dtype=float)
    

    def set_gripper(self, opening):
        self.finger_target = float(opening)
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(
                self.panda,
                i,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=self.finger_target,
                force=self.gripper_force,
            )

    def solve_ik_and_apply(self, target_pos, target_orn):
        current_q = self.get_current_arm_joints()

        jointPoses = self.bullet_client.calculateInverseKinematics(
            self.panda,
            pandaEndEffectorIndex,
            target_pos.tolist(),
            target_orn.tolist(),
            ll, ul, jr, current_q,
            maxNumIterations=50,
        )
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(
                self.panda,
                i,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=jointPoses[i],
                force=self.arm_force,
            )
  
    def setJoint(self, jointPoses):
        for i in range(pandaNumDofs):
          self.bullet_client.setJointMotorControl2(
              self.panda,
              i,
              self.bullet_client.POSITION_CONTROL,
              targetPosition=jointPoses[i],
              force=self.arm_force,
          )
    def get_current_arm_joints(self):
        q = []
        for i in range(pandaNumDofs):
            js = self.bullet_client.getJointState(self.panda, i)
            q.append(js[0])
        return q


    def prepare_state(self, state):
        ee_pos, ee_orn = self.get_ee_pose()

        self.motion_start_pos = ee_pos.copy()
        self.motion_start_orn = ee_orn.copy()
        self.motion_target_pos = ee_pos.copy()
        self.motion_target_orn = ee_orn.copy()

        if state == "home":
            self.motion_target_pos = self.home_ee_pos.copy()
            self.motion_target_orn = self.home_ee_orn.copy()
            self.target_gripper = 0.04
            self.set_gripper(self.target_gripper)

        elif state == "move_pregrasp":
            # 简单 pick-up：从物体上方接近，不再根据 mug tree / rack 计算侧向 approach。
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()

            pregrasp_pos = grasp_pos.copy()
            pregrasp_pos[2] += self.safe_approach

            self.motion_target_pos = pregrasp_pos
            self.motion_target_orn = grasp_orn.copy()

        elif state == "open_gripper":
            self.target_gripper = 0.04

        elif state == "move_grasp":
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()
            self.last_grasp_pose = grasp_pos.copy()
            self.last_grasp_orn = grasp_orn.copy()

            self.motion_target_pos = grasp_pos.copy()
            self.motion_target_orn = grasp_orn.copy()

        elif state == "close_gripper":
            self.target_gripper = 0.0

        elif state == "lift_object":
            # 闭爪后直接沿 world z 方向抬起物体，完成简单 pick-up。
            self.motion_target_pos = ee_pos.copy() + np.array([0.0, 0.0, 0.2], dtype=float)
            self.motion_target_orn = ee_orn.copy()

    def switch_to_next_state(self):
        self.state_idx += 1
        if self.state_idx >= len(self.states):
            self.done = True
            return

        self.state = self.states[self.state_idx]
        self.state_t = 0.0
        self.prepare_state(self.state)
        print("state ->", self.state)

    def step(self):
        self.t += self.control_dt
        self.state_t += self.control_dt

        duration = self.state_durations[self.state_idx]
        s = min(self.state_t / duration, 1.0)

        if self.state in ["open_gripper", "close_gripper"]:
            self.set_gripper(self.target_gripper)

            ee_pos, ee_orn = self.get_ee_pose()
            self.target_pos = ee_pos.copy()
            self.target_orn = ee_orn.copy()

        elif self.state in ["home", "move_pregrasp", "move_grasp", "lift_object"]:
            self.target_pos = (1.0 - s) * self.motion_start_pos + s * self.motion_target_pos
            self.target_orn = quat_slerp(self.motion_start_orn, self.motion_target_orn, s)
            self.solve_ik_and_apply(self.target_pos, self.target_orn)

        if self.state_t >= duration:
            self.switch_to_next_state()

        return self.target_pos, self.target_orn

    def collect_observation(self):
        """
        TODO: Collect observations for diffusion policy
        obs = {
            "agentview_image": (H, W, 3),          # 可选（camera）
            "robot0_eye_in_hand_image": (H, W, 3),
            "robot0_eef_pos": (3,),             # 末端位置 xyz
            "robot0_eef_quat": (4,),            # 末端姿态 quaternion
            "robot0_gripper_qpos": (2,),        # `手爪开合（两个指）
        }
        """
        eef_pos, eef_quat = self.get_ee_pose()

        gripper_qpos = np.array([
            self.bullet_client.getJointState(self.panda, 9)[0],
            self.bullet_client.getJointState(self.panda, 10)[0],
        ], dtype=np.float32)

        _, _, agentview_rgba, _, _ = self.get_agentview_image()
        agentview = agentview_rgba[..., :3]
        eye_in_hand = self.get_eye_in_hand_image()

        obs = {
            "agentview_image": agentview.astype(np.uint8),                  # (H,W,3)
            "robot0_eye_in_hand_image": eye_in_hand.astype(np.uint8),      # (H,W,3)
            "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32),       # (3,)
            "robot0_eef_quat": np.asarray(eef_quat, dtype=np.float32),     # (4,)
            "robot0_gripper_qpos": gripper_qpos,                           # (2,)
        }
        return obs
    
  
    def collect_action(self):      
        # 1维 gripper command，开=0.04，关=0.0
        gripper = np.array([self.target_gripper], dtype=np.float32)

        action = np.concatenate([self.target_pos, self.target_orn, gripper], axis=0).astype(np.float32)
        return action


    def apply_image_noise(self, rgb):
        return self.ImgNoiseDR.apply_image_noise(rgb)


    def render_camera(self, width, height, view_matrix, proj_matrix):
        light_cfg = self.lightingDR.light_cfg

        w, h, rgba, depth, seg = self.bullet_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL,
            lightDirection=light_cfg["lightDirection"],
            lightColor=light_cfg["lightColor"],
            lightDistance=light_cfg["lightDistance"],
            lightAmbientCoeff=light_cfg["lightAmbientCoeff"],
            lightDiffuseCoeff=light_cfg["lightDiffuseCoeff"],
            lightSpecularCoeff=light_cfg["lightSpecularCoeff"],
            shadow=light_cfg["shadow"],
        )

        rgba = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4).copy()
        depth = np.asarray(depth, dtype=np.float32).reshape(h, w)
        seg = np.asarray(seg, dtype=np.int64).reshape(h, w)

        # DIRECT/headless change sky to pure black

        bg_mask = seg < 0
        rgba[bg_mask, :3] = 0
        rgba[bg_mask, 3] = 255

        rgb = rgba[..., :3]
        rgb = self.apply_image_noise(rgb)
        rgba[..., :3] = rgb


        return w, h, rgba, depth, seg


    def center_crop_resize_rgb(self, img, out_size=224):
    # TODO： change to diffusion policy's rescale method
        """
        img: RGB image, shape [H, W, 3]
        return: RGB image, shape [out_size, out_size, 3]
        """
        h, w = img.shape[:2]

        # 先中心裁剪成正方形，避免直接 resize 导致图像变形
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2

        crop = img[y0:y0 + side, x0:x0 + side]

        # 再 resize 到 224x224
        img_224 = cv2.resize(
            crop,
            (out_size, out_size),
            interpolation=cv2.INTER_AREA
        )

        return img_224
    
    def get_cropped_agentview_image(self, out_size=224):
        _, _, rgba, _, _ = self.get_agentview_image()
        rgb = rgba[..., :3]
        cropped_rgb = self.center_crop_resize_rgb(rgb, out_size)
        return cropped_rgb

    def get_agentview_image(self):
        W = 960
        H = 540
        near = 0.02
        far  = 2.0
        extrinsic_cam = np.array([[-0.808, 0.3283, -0.4892,  0.8758,],
                                [ 0.5837,  0.3327, -0.7407,  0.6006,],
                                [-0.0804, -0.884,  -0.4604,  0.4729,],
                                [ 0.,      0.,      0.,      1.,    ],], dtype=np.float32)
        
        intrinsic = np.array([[691.7508,    0.,      486.7637,],
                            [  0.,    692.2195,   273.4784,],
                            [  0.,      0.,       1.,  ],], dtype=np.float32)
        projectionMatrix = cvK2BulletP(intrinsic, W, H, near, far)
        viewMatrix = cvPose2BulletView(extrinsic_cam)

        return self.render_camera(W, H, viewMatrix, projectionMatrix)
  
    def get_eye_in_hand_image(self, width=224, height=224):
        camera_link = 8  # panda_hand

        link_state = self.bullet_client.getLinkState(
            self.panda,
            camera_link,
            computeForwardKinematics=True
        )

        link_pos = np.array(link_state[4], dtype=np.float32)
        link_quat = np.array(link_state[5], dtype=np.float32)

        rot = np.array(
            self.bullet_client.getMatrixFromQuaternion(link_quat),
            dtype=np.float32
        ).reshape(3, 3)

        # camera mounting
        local_offset = np.array([
            0.05,     # 相机在 wrist 上方5cm
            0.0,     
            0.05     # 相机在wrist前方1cm
        ], dtype=np.float32)

        cam_pos = link_pos + rot @ local_offset
        # 相机lookat方向
        cam_forward = rot @ np.array([0, 0, 1], dtype=np.float32)
        cam_up = rot @ np.array([0, -1, 0], dtype=np.float32)

        target = cam_pos + 0.2 * cam_forward

        view = self.bullet_client.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=cam_up.tolist()
        )

        proj = self.bullet_client.computeProjectionMatrixFOV(
            fov= 160.0,
            aspect=float(width) / float(height),
            nearVal=0.001,
            farVal=2.0
        )

        _, _, rgba, _, _ = self.render_camera(width, height, view, proj)
        return rgba[..., :3]
    

    def is_success(self, lift_height_threshold=0.08, require_contact=True):
        if not hasattr(self, "pick_up_obj_id"):
            return False

        obj_pos, _ = get_true_PositionAndOrientation(
            self.bullet_client, self.pick_up_obj_id
        )
        obj_pos = np.array(obj_pos, dtype=float)

        if self.initial_obj_pos is None:
            lifted = obj_pos[2] > lift_height_threshold
        else:
            lifted = (obj_pos[2] - self.initial_obj_pos[2]) > lift_height_threshold

        if not require_contact:
            return bool(lifted)

        contacts = self.bullet_client.getContactPoints(
            bodyA=self.panda,
            bodyB=self.pick_up_obj_id,
        )
        grasped = len(contacts) > 0

        return bool(lifted and grasped)

    def enable_high_quality_rendering(self):
        try:
            self.bullet_client.configureDebugVisualizer(
                self.bullet_client.COV_ENABLE_SHADOWS, 1
            )
            self.bullet_client.configureDebugVisualizer(
                self.bullet_client.COV_ENABLE_RENDERING, 1
            )
            print("Enabled high quality rendering.")
        except Exception:
            pass
