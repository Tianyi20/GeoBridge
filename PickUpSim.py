import time
import gym
import numpy as np
import math
from icecream import ic
import pybullet_data
from utility import load_initial_grasp_pose

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
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

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

        self.panda = self.bullet_client.loadURDF(
            "franka_panda/panda.urdf",
            np.array([0, 0, 0]) + self.offset,
            base_orn,
            useFixedBase=True,
            flags=flags,
        )

        ## Make scene, TODO: randomization
        ## invoke make scene outside
        ## Load initial guess

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

        # ========= 状态机 =========
        self.states = [
            "home",
            "approach_mugtree",
            "approach_manipulated_cup",
            "move_pregrasp",
            "open_gripper",
            "move_grasp",
            "close_gripper",
            "extract_cup",
        ]
        self.state_durations = [0.01, 0.5, 1.0, 0.5, 0.1, 1.0, 0.1, 1.0]
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
                   obj_pose_offset = [0.5, 0.0, 0.0],
                   obj_euler_offset = [0.0, 0.0, 0.0],
                   ):

        self.env_mesh_path = env_mesh_path
        self.pick_up_obj_path = manipulated_obj_path
        self.convex_pick_up_obj_path = coacd_convex_decomposition(self.pick_up_obj_path)
        self.com_pick_up_obj = get_com(self.pick_up_obj_path)

        # TODO: Load ground plane but make it invisible
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
        # TODO: load background, but make it non-collidable
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
            position=np.array(obj_pose_offset),
            baseOrientation=self.bullet_client.getQuaternionFromEuler(np.array(obj_euler_offset)),
            center_of_mass=np.array(self.com_pick_up_obj),
            lateral_friction=0.6,
        )
        self.initial_grasp_guess = load_initial_grasp_pose("grasp_pose.yaml")


        self.waite_scene_stable()

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
                                                                         self.manipulated_cup_id)
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
        
        elif state == "approach_mugtree":
            aabbMin, aabbMax = self.bullet_client.getAABB(self.mug_tree)
            aabbMin = np.array(aabbMin, dtype=float)
            aabbMax = np.array(aabbMax, dtype=float)

            grasp_pos, grasp_orn = self.get_initial_guess_grasp()
            grasp_pos = np.array(grasp_pos, dtype=float)

            # 外侧安全距离：离AABB侧面稍微退开一点
            side_margin = self.safe_approach
            # 顶部抬高距离：放到AABB最上方再高一点
            top_margin = 0.05

            # grasp point 到四个侧面的距离
            dist_to_xmin = abs(grasp_pos[0] - aabbMin[0])
            dist_to_xmax = abs(aabbMax[0] - grasp_pos[0])
            dist_to_ymin = abs(grasp_pos[1] - aabbMin[1])
            dist_to_ymax = abs(aabbMax[1] - grasp_pos[1])

            side_dists = {
                "xmin": dist_to_xmin,
                "xmax": dist_to_xmax,
                "ymin": dist_to_ymin,
                "ymax": dist_to_ymax,
            }

            nearest_face = min(side_dists, key=side_dists.get)

            pregrasp_pos = grasp_pos.copy()

            # 先放到“最近侧面”的外侧
            if nearest_face == "xmin":
                pregrasp_pos[0] = aabbMin[0] - side_margin
            elif nearest_face == "xmax":
                pregrasp_pos[0] = aabbMax[0] + side_margin
            elif nearest_face == "ymin":
                pregrasp_pos[1] = aabbMin[1] - side_margin
            elif nearest_face == "ymax":
                pregrasp_pos[1] = aabbMax[1] + side_margin

            # 再放到 AABB 顶面上方
            pregrasp_pos[2] = aabbMax[2] + top_margin

            self.motion_target_pos = pregrasp_pos
            self.motion_target_orn = self.motion_start_orn
        
        elif state == "approach_manipulated_cup":
            ## get the mug tree center, future grasp pose.
            ## “沿抓取方向后退”的 pregrasp pose 
            tree_center, tree_orn = get_COM_PositionAndOrientation(self.bullet_client, 
                                                                   self.mug_tree)
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()

            radial = grasp_pos - tree_center
            radial = radial / np.linalg.norm(radial)

            up = np.array([0.0, 0.0, 1.0], dtype=float)

            alpha = 0.8
            beta = 0.2
            v = alpha * radial + beta * up
            v = v / np.linalg.norm(v)

            pregrasp_pos = grasp_pos + self.safe_approach * v

            self.motion_target_pos = pregrasp_pos
            self.motion_target_orn = self.motion_start_orn

        elif state == "move_pregrasp":
            cup_base, _ = get_true_PositionAndOrientation(self.bullet_client, 
                                                          self.manipulated_cup_id)
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()

            v = grasp_pos - cup_base
            v = v / np.linalg.norm(v)
            pregrasp_pos = grasp_pos + self.safe_grasp_offset * v

            self.motion_target_pos = pregrasp_pos
            self.motion_target_orn = grasp_orn
        

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

        elif state == "extract_cup":
            #TODO： 现在这个是基于 grasp pose的，最好改成基于mug tree的树枝方向的
            if self.last_grasp_pose is None:
                base_pos, base_orn = self.get_initial_guess_grasp()
            else:
                base_pos = self.last_grasp_pose.copy()
                base_orn = self.last_grasp_orn.copy()

            move_dir_world = np.array(
                self.bullet_client.multiplyTransforms(
                    [0, 0, 0],
                    base_orn.tolist(),
                    [0.0, -1.0, 0.0],   # local -x = local y rotated +90deg around local z
                    [0, 0, 0, 1],
                )[0],
                dtype=float,
            )

            move_dist = 0.2
            target_pos = base_pos.copy() + move_dist * move_dir_world

            self.motion_target_pos = target_pos
            self.motion_target_orn = base_orn.copy()

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

        elif self.state in ["home", "approach_mugtree", "approach_manipulated_cup",
                            "move_pregrasp", "move_grasp", "extract_cup"]:
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

        agentview = self.get_agentview_image()
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


    def render_camera(self, width, height, view_matrix, proj_matrix):
        _, _, rgba, _, _ = self.bullet_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL,
            # lightDirection=[0, -1, 0.8],               
            # lightColor=[1, 1, 1],                   
            # lightDistance=3.0,                      
            # shadow=1,
        )
        rgb = np.asarray(rgba, dtype=np.uint8)[..., :3]
        return rgb

    def get_agentview_image(self):
        # target = [0.4, 0.00, 0.2]
        # dist = 1.5
        # yaw = 90 
        # pitch = -45  
        # roll = 0
        # up_axis = 2
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
        # self.visualize_camera(cam_pos)
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

        return self.render_camera(width, height, view, proj)
    
    def visualize_camera(self, pos_to_visualize):
        """
        use a red sphere to visualize given pose
        """
        radius = 0.01  # 1cm 小球

        visual_shape = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 1]  # 红色
        )

        # 创建一个纯视觉 object（无碰撞）
        self.camera_marker = self.bullet_client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos_to_visualize.tolist()
        )

    def is_success(self, height_threshold=0.1, dist_threshold=0.1):
        """
        判断任务是否成功：
        1. 杯子被抬起（高度变化）
        2. 杯子远离 mug tree
        """

        # 当前杯子位置
        cup_pos, _ = get_true_PositionAndOrientation(
            self.bullet_client, self.manipulated_cup_id
        )

        # mug tree 中心
        tree_pos, _ = get_COM_PositionAndOrientation(
            self.bullet_client, self.mug_tree
        )

        contacts = self.bullet_client.getContactPoints(
            bodyA=self.panda,
            bodyB=self.manipulated_cup_id
        )

        grasped = len(contacts) > 0

        cup_pos = np.array(cup_pos)
        tree_pos = np.array(tree_pos)

        # 条件1：高度变高（被拿起）
        lifted = cup_pos[2] > height_threshold

        # 条件2：远离 rack
        far_enough = np.linalg.norm(cup_pos - tree_pos) > dist_threshold

        return lifted and far_enough and grasped