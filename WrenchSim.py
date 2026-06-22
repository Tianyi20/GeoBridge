import time
import cv2
import gym
import numpy as np
import math
import pybullet_data
from VisualDR import LightingDR, ImgNoiseDR, DistractorDR, PoseDR, ObjectColorDR, FPSAObjectDR
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

from utility import (
    load_initial_grasp_pose, 
    resize_rgb, 
    normalize_vector, 
    quat_from_rotation_matrix,
    shrink_mesh,
    ) 

from icecream import ic
from scipy.spatial.transform import Rotation as R

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
ul = [7] * pandaNumDofs
jr = [7] * pandaNumDofs

jointPositions = [
                0.0,
                -0.74586,
                0.0,
                -2.84895,
                0.0,
                2.10301,
                np.pi / 4,
                0.04,
                0.04,
]
rp = jointPositions


class WrenchSim(object):
    def __init__(self, bullet_client, offset, 
                 control_dt = 1.0 / 120.0, seed = 42):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

        # Visual domain randomizers. Each randomizer owns its own RNG and state.
        # Keep bullet_client inside every module so each multiprocessing worker can
        # create an independent PickUpSim / VisualDR stack.
        self.lightingDR = LightingDR(self.bullet_client, seed=seed)
        self.ImgNoiseDR = ImgNoiseDR(self.bullet_client, seed=seed)
        self.objposeDR  = PoseDR(self.bullet_client, seed=seed)
        self.camposeDR  = PoseDR(self.bullet_client, seed=seed)
        self.outsceneDR = PoseDR(self.bullet_client, seed=seed)
        self.collplaneDR = PoseDR(self.bullet_client, seed=seed)
        self.distractorDR = DistractorDR(self.bullet_client, seed=seed)
        self.objectColorDR = ObjectColorDR(self.bullet_client, seed=seed)
        self.fpsaObjectDR = FPSAObjectDR(seed=seed)

        self.offset = np.array(offset)
        self.control_dt = control_dt
        self.t = 0.0
        # Set GUI's Sky to white, only influence GUI
        self.bullet_client.configureDebugVisualizer(rgbBackground=[1, 1, 1])

        # Based Agent-view camera calibration / pose state.
        self.agentview_width = 960
        self.agentview_height = 540
        self.agentview_near = 0.02
        self.agentview_far = 2.0
        self.agentview_base_extrinsic_cam = np.array([
            [-0.808,   0.3283, -0.4892,  0.8758],
            [ 0.5837,  0.3327, -0.7407,  0.6006],
            [-0.0804, -0.884,  -0.4604,  0.4729],
            [ 0.0,     0.0,     0.0,     1.0   ],
        ], dtype=np.float32)
        self.agentview_base_intrinsic = np.array([
            [691.7508,   0.0,     486.7637],
            [  0.0,    692.2195, 273.4784],
            [  0.0,      0.0,       1.0   ],
        ], dtype=np.float32)

        # Gripper command convention for policy / real robot interface:
        #   1.0 = open, 0.0 = closed.
        # PyBullet still needs a physical finger width, so the binary state is
        # converted to width only inside the simulator control layer.
        self.GRIPPER_CLOSED = 0.0
        self.GRIPPER_OPEN = 1.0
        self.gripper_closed_width = 0.0
        self.gripper_open_width = 0.04
        self.gripper_state_threshold_width = 0.8 * (
            self.gripper_open_width + self.gripper_closed_width
        )
        self.target_gripper = self.GRIPPER_OPEN
        self.finger_target = self.gripper_open_width

        # Fixed grasp orientation convention:
        # local +Z of the end-effector is treated as the positive grasp normal.
        # It is locked to a world-frame normal, so object yaw randomization
        # (e.g. banana rotated by 180 deg) will not flip the gripper angle.
        # Change these two vectors if the real gripper frame uses a different
        # positive normal / in-plane tangent convention.
        self.grasp_world_normal = np.array([0.0, 0.0, -1.0], dtype=float)
        self.grasp_world_tangent = np.array([1.0, 0.0, 0.0], dtype=float)

        self.safe_approach = 0.15
        self.safe_grasp_offset = 0.05
        self.arm_force = 200.0
        self.gripper_force = 100.0

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        base_orn = self.bullet_client.getQuaternionFromEuler([0, 0, 0])

        ################## Load Panda robot #################
        self.panda = self.bullet_client.loadURDF(
            "franka_panda/panda_wristcam.urdf",
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

        #########========= wrench screw engagement / fastening 状态机 =========###################
        self.states = [
            "home",
            "move_preEngage",
            "move_Engage",
            "fasten_screw",
        ]
        self.state_durations = [0.01, 4.0, 4.0, 4.0]

        # fasten_screw: keep current wrench TCP position fixed and rotate
        # the TCP frame about its own local +Z axis by this angle.
        self.fasten_angle_rad = -math.radians(45.0)
        self.state_idx = 0
        self.state = self.states[self.state_idx]
        self.state_t = 0.0

        # target pose, target gripper state (1=open, 0=closed)
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

    def load_wrench_tool(self):
        """Load wrench mesh as a separate PyBullet body and fix it to the robot EE link."""
        p = self.bullet_client

        # 推荐 collision 用 COACD/VHACD 后的 convex mesh
        self.wrench_collision_asset = shrink_mesh(
            self.wrench_mesh_path,
            shrink_mode="by_margin",
            margin_m=0.005,
            shrink_center="centroid",
        )
        wrench_collision_path = coacd_convex_decomposition(self.wrench_collision_asset)

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=self.wrench_mesh_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.8, 0.8, 0.8, 1.0],
        )

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=wrench_collision_path,
            meshScale=[1.0, 1.0, 1.0],
        )

        parent_state = p.getLinkState(
            self.panda,
            self.tool_parent_link,
            computeForwardKinematics=True,
        )
        parent_pos = np.array(parent_state[4], dtype=float)
        parent_orn = np.array(parent_state[5], dtype=float)

        wrench_pos, wrench_orn = p.multiplyTransforms(
            parent_pos.tolist(),
            parent_orn.tolist(),
            self.parent_to_wrench_pos.tolist(),
            self.parent_to_wrench_orn.tolist(),
        )

        self.wrench_body_id = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=wrench_pos,
            baseOrientation=wrench_orn,
        )

        p.changeDynamics(
            self.wrench_body_id,
            -1,
            lateralFriction=0.7,
            spinningFriction=0.00,
            rollingFriction=0.005,
        )

        self.wrench_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.panda,
            parentLinkIndex=self.tool_parent_link,
            childBodyUniqueId=self.wrench_body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=self.parent_to_wrench_pos.tolist(),
            parentFrameOrientation=self.parent_to_wrench_orn.tolist(),
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1],
        )

        p.changeConstraint(
            self.wrench_constraint_id,
            maxForce=1000,
            erp=0.9,
        )

        # 关掉 wrench 和 robot 自身碰撞，避免工具和手爪/手腕打架
        for link_idx in range(-1, p.getNumJoints(self.panda)):
            p.setCollisionFilterPair(
                self.panda,
                self.wrench_body_id,
                link_idx,
                -1,
                enableCollision=0,
            )
        # set gripper width
        self.set_gripper(0.0125)

    def get_parent_ee_pose(self):
        p = self.bullet_client
        link_state = p.getLinkState(
            self.panda,
            self.tool_parent_link,
            computeForwardKinematics=True,
        )
        parent_pos = np.array(link_state[4], dtype=float)
        parent_orn = np.array(link_state[5], dtype=float)
        return parent_pos, parent_orn


    def tcp_pose_to_parent_pose(self, tcp_pos, tcp_orn):
        p = self.bullet_client

        inv_tcp_pos, inv_tcp_orn = p.invertTransform(
            self.parent_to_tcp_pos.tolist(),
            self.parent_to_tcp_orn.tolist(),
        )

        parent_pos, parent_orn = p.multiplyTransforms(
            np.asarray(tcp_pos, dtype=float).tolist(),
            np.asarray(tcp_orn, dtype=float).tolist(),
            inv_tcp_pos,
            inv_tcp_orn,
        )

        return np.array(parent_pos, dtype=float), np.array(parent_orn, dtype=float)

    def make_scene(self, 
                   env_mesh_path        = None,
                   manipulated_obj_path = None,
                   manipulated_obj_collision_path = None,
                   clipper_obj_path     = None,
                   initial_grasp_path = None,
                   if_FPSA = False,
                   fpsa_aug_root = "~/GeoBridge/data/objects/bracket/fpsa_aug_outputs",
                   fpsa_include_base = True,
                   obj_pose_base = [0.5, 0.0, 0.0],
                   obj_euler_base = [0.0, 0.0, 0.0],
                   randomize_lighting = True,
                   # outlier scene         
                   randomize_outlscene  = True,
                   outlscene_xyz_jit    = 0.02,
                   outlscene_eul_jit    = 0.01,
                   # plane height randomization
                   randomize_plane_height = True,
                   plane_height_jit = 0.008,
                   randomize_objpose  = True,
                   obj_x_jit    = 0.2,
                   obj_y_jit    = 0.2,
                   obj_z_jit    = 0.1,
                   obj_z_eul_jit = np.pi,
                   randomize_campose = True,
                   cam_xyz_jit  = 0.004,
                   cam_eul_jit  = 0.002,
                   randomize_image_noise = True,
                   # manipulated object color / material randomization
                   randomize_object_color = True,
                   object_color_mode = "bounded",  # "bounded" or "recolor"
                   object_color_strength = 0.35,
                   object_recolor_palette = None,
                   object_recolor_target_color = None,
                   object_specular_range = (0.02, 0.8),
                   randomize_distractors = True,
                   distractor_root = "/mnt/storage/GoogleScannedObjects",
                   distractor_num_range = (1, 5),
                   distractor_target_size_range = (0.06, 0.16),
                   distractor_workspace = ((0.25, 0.78), (-0.42, 0.42)),
                   distractor_clearance = 0.035,
                   distractor_path_clearance = 0.04,
                   distractor_min_target_mask_pixels = 1,
                   ):
        
        ## TODO: visual things to be randomized:
        # task wrench engagement
        # 1. wrench tool assembling


        # always enable high quality rendering pipeline and shadows

        """ ################ Load basic scene assets ################ """
        # Avoid mutating caller-provided lists/default arguments when pose randomization is enabled.
        self.use_wrench_tcp = True
        self.wrench_body_id = None
        self.wrench_constraint_id = None

        # wrench mesh
        self.wrench_mesh_path = "/home/iadc/GeoBridge/data/objects/wrench/wrench_repaired.obj"

        # 工具固定在哪个 robot link 上
        # 先用你当前的 EE link；如果发现不对，再换成 panda_hand 的 index，比如 8
        self.tool_parent_link = pandaEndEffectorIndex

        # parent link frame -> wrench mesh origin
        # 这个需要你根据 mesh 在 Blender / MeshLab / SolidWorks 里调
        self.parent_to_wrench_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.parent_to_wrench_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # parent link frame -> wrench TCP
        # 这个才是以后 policy/action/observation 用的 TCP
        # here our base TCP relative to mesh origin is 0.16417
        self.parent_to_tcp_pos = np.array([0.16417, 0.0, 0.0], dtype=float)
        self.parent_to_tcp_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # load wrench tool
        self.load_wrench_tool()

        # After enabling the wrench TCP, refresh the home pose so all state-machine
        # targets are expressed in the same TCP frame used by get_ee_pose() and
        # solve_ik_and_apply().
        self.home_ee_pos, self.home_ee_orn = self.get_ee_pose()
        self.home_ee_pos = np.array(self.home_ee_pos, dtype=float)
        self.home_ee_orn = np.array(self.home_ee_orn, dtype=float)
        if self.state == "home":
            self.prepare_state(self.state)

        obj_pose_base = np.array(obj_pose_base, dtype=float).copy()
        obj_euler_base = np.array(obj_euler_base, dtype=float).copy()

        if if_FPSA:
            manipulated_obj_path, initial_grasp_path = self.fpsaObjectDR.sample(
                base_mesh_path=manipulated_obj_path,
                base_grasp_path=initial_grasp_path,
                fpsa_aug_root=fpsa_aug_root,
                include_base=fpsa_include_base,
            )
            self.fpsa_object_sample = self.fpsaObjectDR.last_sample
        else:
            self.fpsa_object_sample = None

        self.env_mesh_path = env_mesh_path
        self.screw_obj_path = manipulated_obj_path
        self.clipper_obj_path = clipper_obj_path
        self.screw_collision_asset = manipulated_obj_collision_path

        self.convex_clipper_path = coacd_convex_decomposition(self.clipper_obj_path)
        self.screw_collision_path = shrink_mesh(
            self.screw_collision_asset,
            shrink_mode="by_margin",
            margin_m=0.003,
            shrink_center="centroid",
        )
        self.convex_screw_obj_path = coacd_convex_decomposition(self.screw_collision_path)
        self.com_screw = get_com(self.screw_obj_path)

        self.initial_grasp_path = initial_grasp_path

        """ ################ Basic Visual Randomization ############### """
        if randomize_lighting:
            self.lightingDR.sample_lighting_randomization()
        else:
            self.lightingDR.reset_to_default()
        
        if randomize_objpose:
            objPose, objOrn = self.objposeDR.sample_SE3_randomization(
                pos=obj_pose_base,
                orn=self.bullet_client.getQuaternionFromEuler(np.array(obj_euler_base)),
                x_jitter_range=obj_x_jit,
                y_jitter_range=obj_y_jit,
                z_jitter_range=obj_z_jit,
                z_euler_jitter_range=obj_z_eul_jit,
            )
        else:
            objPose = obj_pose_base
            objOrn = self.bullet_client.getQuaternionFromEuler(np.array(obj_euler_base))

        if randomize_image_noise:
            self.ImgNoiseDR.sample_image_noise_randomization()
        else:
            self.ImgNoiseDR.reset()

        # Background scene mesh pose and invisible collision-plane height.
        #
        # The visual scene mesh gets its own pose randomization first.
        # Then the collision plane follows the randomized mesh z and receives
        # an extra height jitter:
        #
        #   mesh_z  = mesh_z_base
        #   plane_z = mesh_z_base + ground_z_jit
        outscene_base_pos = np.array([0.0, 0.0, -0.005], dtype=float)
        outscene_base_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        if randomize_outlscene:
            env_mesh_pos, env_mesh_orn = self.outsceneDR.sample_SE3_randomization(
                pos=outscene_base_pos,
                orn=outscene_base_orn,
                x_jitter_range=outlscene_xyz_jit,
                y_jitter_range=outlscene_xyz_jit,
                z_jitter_range=outlscene_xyz_jit,
                x_euler_jitter_range= None,
                y_euler_jitter_range= None,
                z_euler_jitter_range=outlscene_eul_jit,
            )
        else:
            env_mesh_pos = outscene_base_pos.copy()
            env_mesh_orn = outscene_base_orn.copy()

        # collision plane height
        if randomize_plane_height:
            ground_pose = self.collplaneDR.sample_pos_randomization(
                pos= [0.0, 0.0, env_mesh_pos[2]],
                z_jitter_range= plane_height_jit,
            )
        else:
            ground_pose = np.array([0.0, 0.0, env_mesh_pos[2]], dtype=np.float32)
        # Load ground plane but make it invisible.
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_plane_id = self.bullet_client.loadURDF(
            "plane.urdf",
            basePosition=ground_pose,
        )
        # Make the plane invisible but keep collision.
        self.bullet_client.changeVisualShape(
            self.ground_plane_id,
            -1,
            rgbaColor=[1, 1, 1, 0],  # alpha = 0
        )

        # camera pose randomization
        if randomize_campose:
            self.extrinsic_cam = self.camposeDR.sample_SE3_randomization(
                pos=self.agentview_base_extrinsic_cam[:3, 3],
                orn=quat_from_rotation_matrix(self.agentview_base_extrinsic_cam[:3, :3]),
                x_jitter_range=cam_xyz_jit,
                y_jitter_range=cam_xyz_jit,
                z_jitter_range=cam_xyz_jit,
                x_euler_jitter_range=cam_eul_jit,
                y_euler_jitter_range=cam_eul_jit,
                z_euler_jitter_range=cam_eul_jit,
                get_matrix=True,
            )
        else:
            self.extrinsic_cam = self.agentview_base_extrinsic_cam.copy()

        # Load background / outlier scene as visual-only.
        self.env_mesh = load_models(
            self.bullet_client,
            visual_mesh_file=self.env_mesh_path,
            vhacd_mesh_file=None,
            desired_mass=0.0,
            position=env_mesh_pos,
            baseOrientation=env_mesh_orn,
            visual_only=True,
        )

        self.screw_obj_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.screw_obj_path,
            vhacd_mesh_file=self.convex_screw_obj_path,
            desired_mass=0.1,
            position= objPose,
            baseOrientation= objOrn,
            center_of_mass=np.array(self.com_screw),
            lateral_friction=0.09,
            spinning_friction= 0.00,
        )

        # Load the clipper that support screw
        # The position relative to screw position should be fixed
        self.clipper_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.clipper_obj_path,
            vhacd_mesh_file=self.convex_clipper_path,
            desired_mass=0.0, # static obj with mass = 0.0
            position= objPose + np.array([0.0, 0.0, -0.011]),
            baseOrientation= np.array([0.0, 0.0, 0.7071067811865475, 0.7071067811865476]),
            center_of_mass=np.array(self.com_screw),
            lateral_friction=0.1,
            spinning_friction= 0.0,
        )

        # time.sleep(1.0)
        # Object-level color / material domain randomization.
        # This is applied after the target object has been loaded into PyBullet.
        # It reads the current visual rgbaColor as the original/base color, then
        # applies a global tint through changeVisualShape(). For textured OBJ
        # assets, the PNG texture is not edited/replaced; the tint is applied on
        # top of the existing visual material, so texture details can remain.
        if randomize_object_color:
            self.object_color_cfg = self.objectColorDR.sample_and_apply_object_color_randomization(
                body_id=self.screw_obj_id,
                mode=object_color_mode,
                strength=object_color_strength,
                recolor_palette=object_recolor_palette,
                recolor_target_color=object_recolor_target_color,
                specular_range=object_specular_range,
                alpha=None,  # preserve current visual alpha
            )
        else:
            self.objectColorDR.reset()
            self.object_color_cfg = None

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
                target_body_id=self.screw_obj_id,
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
                # Ground-relative distractor injection. The collision plane z can be randomized;
                # distractors should spawn relative to the actual current ground plane, not world z=0.
                ground_z=float(ground_pose[2]),
                spawn_clearance=0.005,
                # Keep this false by default. Full robot-plan IK rejection is very sensitive to
                # randomized scene/ground pose and can reject almost every candidate.
                check_robot_plan=False,
                check_xy_safety=True,
                min_visible_fraction=0.55,
                debug=False,
            )
        else:
            self.distractorDR.clear_distractors()

        # Step a few frames after static distractor creation so broad-phase contacts
        # are updated before collecting observations.
        for _ in range(5):
            self.bullet_client.stepSimulation()

        obj_pos, obj_orn = get_true_PositionAndOrientation(
            self.bullet_client, self.screw_obj_id
        )
        self.initial_obj_pos = np.array(obj_pos, dtype=float)
        self.initial_obj_orn = np.array(obj_orn, dtype=float)

    def get_state_machine_ee_waypoints(self):
        """Approximate the wrench TCP path used by the screw-engagement state machine."""
        engage_pos, engage_orn = self.get_initial_guess_grasp()

        # Move 10 cm along the engaging TCP pose's local -X direction.
        # This is pose-relative, not global/world -X.
        preengage_pos = self.offset_pos_along_local_axis(
            engage_pos,
            engage_orn,
            local_axis=np.array([1.0, 0.0, 0.0], dtype=float),
            distance=-self.safe_approach,
        )

        fasten_orn = self.rotate_quat_about_local_z(
            engage_orn,
            self.fasten_angle_rad,
        )

        return [
            (self.home_ee_pos.copy(), self.home_ee_orn.copy()),
            (preengage_pos, engage_orn.copy()),
            (engage_pos.copy(), engage_orn.copy()),
            (engage_pos.copy(), fasten_orn.copy()),
        ]

    def waite_scene_stable(self, waite_steps=1000, vel_threshold=0.005):       
        steps = 0 
        while steps < waite_steps:
            self.bullet_client.stepSimulation()
            steps += 1
            vel1, ang_vel1 = self.bullet_client.getBaseVelocity(self.screw_obj_id)

            speed1 = np.linalg.norm(vel1) + np.linalg.norm(ang_vel1)

            if speed1 < vel_threshold:
                print("Scene stabilized.")
                return True

        print("Warning: Scene did not stabilize within timeout.")
        return False



    def get_fixed_normal_grasp_orn(self, raw_grasp_orn=None):
        """Fix only the grasp positive normal direction in world frame.

        The EE local +Z axis is forced to align with self.grasp_world_normal.
        The in-plane x axis is taken from the original annotated grasp orientation,
        then canonicalized to avoid 180-degree flips.
        """
        z_axis = normalize_vector(self.grasp_world_normal)

        # Use the original grasp orientation to choose the in-plane direction.
        # This preserves some information from the manually annotated grasp,
        # instead of fully hard-coding the whole orientation.
        if raw_grasp_orn is not None:
            raw_rot = np.array(
                self.bullet_client.getMatrixFromQuaternion(raw_grasp_orn),
                dtype=float,
            ).reshape(3, 3)

            # EE local +X axis from the original grasp orientation.
            tangent = raw_rot[:, 0]
        else:
            tangent = np.asarray(self.grasp_world_tangent, dtype=float)

        # Project tangent onto the plane perpendicular to the fixed normal.
        tangent = tangent - np.dot(tangent, z_axis) * z_axis

        if np.linalg.norm(tangent) < 1e-8:
            tangent = np.asarray(self.grasp_world_tangent, dtype=float)
            tangent = tangent - np.dot(tangent, z_axis) * z_axis

        if np.linalg.norm(tangent) < 1e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(fallback, z_axis)) > 0.95:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            tangent = fallback - np.dot(fallback, z_axis) * z_axis

        x_axis = normalize_vector(tangent)

        # Canonicalize the x direction.
        # For a parallel gripper, +x and -x are often physically equivalent,
        # but they create a 180-degree quaternion/action jump.
        ref = np.asarray(self.grasp_world_tangent, dtype=float)
        ref = ref - np.dot(ref, z_axis) * z_axis

        if np.linalg.norm(ref) > 1e-8:
            ref = normalize_vector(ref)
            if np.dot(x_axis, ref) < 0.0:
                x_axis = -x_axis

        y_axis = normalize_vector(np.cross(z_axis, x_axis))
        x_axis = normalize_vector(np.cross(y_axis, z_axis))

        # Columns are EE local x/y/z axes expressed in world frame.
        rot = np.column_stack([x_axis, y_axis, z_axis])
        return quat_from_rotation_matrix(rot)


    def get_initial_guess_grasp(self):
        mesh_world_pos, mesh_world_orn = get_true_PositionAndOrientation(
            self.bullet_client,
            self.screw_obj_id,
        )

        grasp_pose, raw_grasp_orn = self.bullet_client.multiplyTransforms(
            mesh_world_pos,
            mesh_world_orn,
            self.initial_grasp_guess["t"],
            self.initial_grasp_guess["quat"],
        )

        # Position follows the object-frame annotated grasp point.
        # Orientation fixes only the grasp normal, not the entire orientation.
        grasp_orn = self.get_fixed_normal_grasp_orn(raw_grasp_orn)

        return np.array(grasp_pose, dtype=float), np.array(grasp_orn, dtype=float)
    
    def get_ee_pose(self):
        p = self.bullet_client

        link_state = p.getLinkState(
            self.panda,
            self.tool_parent_link if getattr(self, "use_wrench_tcp", False) else pandaEndEffectorIndex,
            computeForwardKinematics=True,
        )

        parent_pos = np.array(link_state[4], dtype=float)
        parent_orn = np.array(link_state[5], dtype=float)

        if getattr(self, "use_wrench_tcp", False):
            tcp_pos, tcp_orn = p.multiplyTransforms(
                parent_pos.tolist(),
                parent_orn.tolist(),
                self.parent_to_tcp_pos.tolist(),
                self.parent_to_tcp_orn.tolist(),
            )
            return np.array(tcp_pos, dtype=float), np.array(tcp_orn, dtype=float)

        return parent_pos, parent_orn

    def gripper_state_to_width(self, state):
        """Map binary gripper state to simulated Panda finger width."""
        state = float(state)
        return self.gripper_open_width if state >= 0.5 else self.gripper_closed_width

    def gripper_width_to_state(self, width):
        """Map measured simulated finger width back to binary gripper state."""
        width = float(width)
        return self.GRIPPER_OPEN if width >= self.gripper_state_threshold_width else self.GRIPPER_CLOSED

    def set_gripper_width(self, opening):
        """Low-level simulator command. Policy/action should not call this directly."""
        self.finger_target = float(opening)
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(
                self.panda,
                i,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=self.finger_target,
                force=self.gripper_force,
            )

    def set_gripper_state(self, state):
        """High-level binary gripper command: 1=open, 0=closed."""
        self.target_gripper = self.GRIPPER_OPEN if float(state) >= 0.5 else self.GRIPPER_CLOSED
        self.set_gripper_width(self.gripper_state_to_width(self.target_gripper))

    def set_gripper(self, command):
        """Compatibility wrapper.

        Accepts the new binary command 0/1 and the old width command such as
        0.04. New policy/action code should call set_gripper_state() directly.
        """
        command = float(command)
        if command in (self.GRIPPER_CLOSED, self.GRIPPER_OPEN):
            self.set_gripper_state(command)
        else:
            self.set_gripper_width(command)

    def solve_ik_and_apply(self, target_pos, target_orn, input_frame="wrench_tcp"):
        """Solve IK and apply arm joint commands.

        Args:
            target_pos: Desired target position in world frame.
            target_orn: Desired target quaternion [x, y, z, w] in world frame.
            input_frame:
                - "wrench_tcp" / "tool_tcp": default, old behavior.
                  target_pos/target_orn are interpreted as the desired wrench TCP pose.
                  If use_wrench_tcp=True, the target is converted to the parent/original
                  EE link pose before solving IK.
                - "parent_tcp" / "parent_ee" / "ee" / "parent": target_pos/target_orn
                  are interpreted as the desired parent/original EE link pose directly.
                  No parent_to_tcp inverse transform is applied. This is the interface
                  to use when policy actions are saved in the original EE frame.
        """
        p = self.bullet_client
        current_q = self.get_current_arm_joints()

        input_frame = str(input_frame).lower()
        wrench_tcp_frames = {"wrench_tcp", "tool_tcp", "tcp"}
        parent_frames = {"parent_tcp", "parent_ee", "ee", "parent", "original_ee"}

        ik_target_pos = np.array(target_pos, dtype=float)
        ik_target_orn = np.array(target_orn, dtype=float)

        # When the wrench tool is enabled, IK is solved on the parent/original EE link.
        # Otherwise, fall back to the normal Panda end-effector index.
        ik_link_index = (
            self.tool_parent_link
            if getattr(self, "use_wrench_tcp", False)
            else pandaEndEffectorIndex
        )

        if input_frame in wrench_tcp_frames:
            if getattr(self, "use_wrench_tcp", False):
                # target_pos/target_orn is desired wrench TCP pose.
                # parent_to_tcp is parent/original EE link -> wrench TCP.
                # Therefore desired parent pose = desired TCP pose * inverse(parent_to_tcp).
                inv_tcp_pos, inv_tcp_orn = p.invertTransform(
                    self.parent_to_tcp_pos.tolist(),
                    self.parent_to_tcp_orn.tolist(),
                )

                ik_target_pos, ik_target_orn = p.multiplyTransforms(
                    ik_target_pos.tolist(),
                    ik_target_orn.tolist(),
                    inv_tcp_pos,
                    inv_tcp_orn,
                )

                ik_target_pos = np.array(ik_target_pos, dtype=float)
                ik_target_orn = np.array(ik_target_orn, dtype=float)

        elif input_frame in parent_frames:
            # target_pos/target_orn is already the desired parent/original EE pose.
            # Do not compensate parent_to_tcp again.
            pass

        else:
            raise ValueError(
                f"Unknown input_frame={input_frame!r}. "
                f"Use one of {sorted(wrench_tcp_frames | parent_frames)}."
            )

        jointPoses = p.calculateInverseKinematics(
            self.panda,
            ik_link_index,
            ik_target_pos.tolist(),
            ik_target_orn.tolist(),
            ll, ul, jr, current_q,
            maxNumIterations=50,
        )

        for i in range(pandaNumDofs):
            p.setJointMotorControl2(
                self.panda,
                i,
                p.POSITION_CONTROL,
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


    def offset_pos_along_local_axis(self, pos, quat_xyzw, local_axis, distance):
        """Offset a world-frame position along an axis defined in the pose local frame.

        For pre-engage, use local_axis=[1, 0, 0] and distance=-0.1 to move
        10 cm along the engaging TCP pose's local -X direction, rather than
        subtracting from the global/world x coordinate.
        """
        pos = np.asarray(pos, dtype=float)
        quat_xyzw = np.asarray(quat_xyzw, dtype=float)
        quat_norm = np.linalg.norm(quat_xyzw)
        if quat_norm < 1e-12:
            raise ValueError("Invalid zero-norm quaternion for local-axis offset.")
        quat_xyzw = quat_xyzw / quat_norm

        local_axis = np.asarray(local_axis, dtype=float)
        axis_norm = np.linalg.norm(local_axis)
        if axis_norm < 1e-12:
            raise ValueError("Invalid zero-norm local axis for local-axis offset.")
        local_axis = local_axis / axis_norm

        axis_world = R.from_quat(quat_xyzw).apply(local_axis)
        axis_world = axis_world / np.linalg.norm(axis_world)
        return pos + float(distance) * axis_world

    def rotate_quat_about_local_z(self, quat_xyzw, angle_rad):
        """Rotate a TCP orientation about its own local +Z axis.

        PyBullet and scipy both use quaternion order [x, y, z, w].
        Multiplying current_rot * delta_z means the delta is applied in the
        current TCP local frame, so the world-space rotation axis is the
        current TCP z-axis.
        """
        quat_xyzw = np.asarray(quat_xyzw, dtype=float)
        quat_norm = np.linalg.norm(quat_xyzw)
        if quat_norm < 1e-12:
            raise ValueError("Invalid zero-norm quaternion for fasten_screw.")
        quat_xyzw = quat_xyzw / quat_norm

        current_rot = R.from_quat(quat_xyzw)
        delta_rot = R.from_rotvec(np.array([0.0, 0.0, angle_rad], dtype=float))
        target_quat = (current_rot * delta_rot).as_quat()
        target_quat = target_quat / np.linalg.norm(target_quat)
        return target_quat.astype(float)

    def prepare_state(self, state):
        ee_pos, ee_orn = self.get_ee_pose()

        self.motion_start_pos = ee_pos.copy()
        self.motion_start_orn = ee_orn.copy()
        self.motion_target_pos = ee_pos.copy()
        self.motion_target_orn = ee_orn.copy()

        # Wrench is fixed to the robot, so gripper command is only kept for
        # action/observation compatibility with the previous pick-up pipeline.
        if self.target_gripper is None:
            self.target_gripper = self.GRIPPER_OPEN

        if state == "home":
            self.motion_target_pos = self.home_ee_pos.copy()
            self.motion_target_orn = self.home_ee_orn.copy()
            # self.target_gripper = self.GRIPPER_OPEN
            # self.set_gripper_state(self.target_gripper)

        elif state == "move_preEngage":
            # Move to a safe pose 10 cm along the engaging TCP pose's local -X axis.
            # Do not subtract from global/world x, because the engage pose may be rotated.
            engage_pos, engage_orn = self.get_initial_guess_grasp()

            preengage_pos = self.offset_pos_along_local_axis(
                engage_pos,
                engage_orn,
                local_axis=np.array([1.0, 0.0, 0.0], dtype=float),
                distance=-self.safe_approach,
            )

            self.motion_target_pos = preengage_pos
            self.motion_target_orn = engage_orn.copy()

        elif state == "move_Engage":
            # Same motion logic as the previous move_grasp state:
            # move the wrench TCP to the annotated screw engagement pose.
            engage_pos, engage_orn = self.get_initial_guess_grasp()
            self.last_grasp_pose = engage_pos.copy()
            self.last_grasp_orn = engage_orn.copy()

            self.motion_target_pos = engage_pos.copy()
            self.motion_target_orn = engage_orn.copy()

        elif state == "fasten_screw":
            # Keep the current wrench TCP origin fixed and rotate the TCP frame
            # about its current local +Z axis. Since get_ee_pose() returns the
            # wrench TCP pose and solve_ik_and_apply() expects a desired TCP pose,
            # no extra parent/wrench offset compensation is needed here.
            self.motion_target_pos = ee_pos.copy()
            self.motion_target_orn = self.rotate_quat_about_local_z(
                ee_orn,
                self.fasten_angle_rad,
            )

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

        if self.state in ["home", "move_preEngage", "move_Engage", "fasten_screw"]:
            self.target_pos = (1.0 - s) * self.motion_start_pos + s * self.motion_target_pos
            self.target_orn = quat_slerp(self.motion_start_orn, self.motion_target_orn, s)
            self.solve_ik_and_apply(self.target_pos, self.target_orn)

        if self.state_t >= duration:
            self.switch_to_next_state()

        return self.target_pos, self.target_orn

    def collect_observation(self, direct = False, collect_wrench_ee = False):
        """
        TODO: Collect observations for diffusion policy
        obs = {
            "agentview_image": (H, W, 3),          # 可选（camera）
            "robot0_eye_in_hand_image": (H, W, 3),
            "robot0_eef_pos": (3,),             # 末端位置 xyz
            "robot0_eef_quat": (4,),            # 末端姿态 quaternion
            "robot0_gripper_qpos": (1,),        # 二值手爪状态：1=open, 0=closed
        }
        Use origional franka ee pose instead of new wrench's pose
        """
        if collect_wrench_ee:
            eef_pos, eef_quat = self.get_ee_pose()
        else:
            eef_pos, eef_quat = self.get_parent_ee_pose()

        finger_widths = np.array([
            self.bullet_client.getJointState(self.panda, 9)[0],
            self.bullet_client.getJointState(self.panda, 10)[0],
        ], dtype=np.float32)
        gripper_state = np.array([
            self.gripper_width_to_state(np.mean(finger_widths))
        ], dtype=np.float32)

        # direct get 224 by approximate intrinsic
        # else get 960 540 then rescale 
        if direct: 
            agentview224 = self.direct_get_agent_view()
        else:
            _, _, agentview_rgba, _, _ = self.get_agentview_image()
            agentview = agentview_rgba[..., :3]
            agentview224 = resize_rgb(agentview, out_size=224)
        # eye_in_hand = self.get_eye_in_hand_image()

        obs = {
            "agentview_image": agentview224.astype(np.uint8),                  # (H,W,3)
            # "robot0_eye_in_hand_image": eye_in_hand.astype(np.uint8),      # (H,W,3)
            # pickup only use agent view
            "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32),       # (3,)
            "robot0_eef_quat": np.asarray(eef_quat, dtype=np.float32),     # (4,)
            "robot0_gripper_qpos": gripper_state,                          # (1,), 1=open, 0=closed
        }
        return obs
    
  
    def collect_action(self):      
        "Always collect original ee tcp instead of wrench tcp"
        # 1维 binary gripper command：1=open, 0=closed
        gripper = np.array([self.target_gripper], dtype=np.float32)

        parent_target_pos, parent_target_orn = self.tcp_pose_to_parent_pose(
            self.target_pos,
            self.target_orn,
        )

        action = np.concatenate(
            [parent_target_pos, parent_target_orn, gripper],
            axis=0,
        ).astype(np.float32)

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

    def get_cropped_agentview_image(self, out_size=224):
        _, _, rgba, _, _ = self.get_agentview_image()
        rgb = rgba[..., :3]
        cropped_rgb = resize_rgb(rgb, out_size)
        return cropped_rgb

    def get_agentview_image(self):
        projectionMatrix = cvK2BulletP(self.agentview_base_intrinsic, 
                                       self.agentview_width, self.agentview_height, 
                                       self.agentview_near, self.agentview_far)
        
        viewMatrix = cvPose2BulletView(self.extrinsic_cam)

        return self.render_camera(self.agentview_width, self.agentview_height, viewMatrix, projectionMatrix)

    def direct_get_agent_view(self, out_size=224):
        """
        Directly render agent view at out_size x out_size.

        This approximates:
            get_agentview_image() at 960x540
            -> cv2.resize(..., (out_size, out_size), interpolation=cv2.INTER_AREA)

        Because your resize_rgb directly stretches 960x540 to 224x224,
        we scale fx, cx by out_size / 960 and fy, cy by out_size / 540.
        """
        W0 = self.agentview_width
        H0 = self.agentview_height

        W = out_size
        H = out_size

        near = self.agentview_near
        far = self.agentview_far

        extrinsic_cam = self.extrinsic_cam.copy()
        intrinsic_960x540 = self.agentview_base_intrinsic.copy()

        sx = W / W0
        sy = H / H0

        intrinsic = intrinsic_960x540.copy()
        intrinsic[0, 0] *= sx  # fx
        intrinsic[0, 2] *= sx  # cx
        intrinsic[1, 1] *= sy  # fy
        intrinsic[1, 2] *= sy  # cy

        projectionMatrix = cvK2BulletP(intrinsic, W, H, near, far)
        viewMatrix = cvPose2BulletView(extrinsic_cam)

        _, _, rgba, _, _ = self.render_camera(W, H, viewMatrix, projectionMatrix)

        rgb = rgba[..., :3]
        return rgb.astype(np.uint8)

  
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
    

    def get_screw_rotation_about_initial_z(self):
        """Return screw yaw/twist change about its initial local z axis in radians.

        The angle is measured by projecting the screw's initial and current
        local +X axes onto the plane perpendicular to the initial local +Z axis,
        then taking the signed angle between the projected axes. This is more
        stable than directly reading Euler yaw when the screw has small tilt.
        """
        if not hasattr(self, "screw_obj_id") or self.initial_obj_orn is None:
            return 0.0

        _, current_obj_orn = get_true_PositionAndOrientation(
            self.bullet_client,
            self.screw_obj_id,
        )

        initial_rot = R.from_quat(np.asarray(self.initial_obj_orn, dtype=float))
        current_rot = R.from_quat(np.asarray(current_obj_orn, dtype=float))

        initial_z_axis = initial_rot.apply(np.array([0.0, 0.0, 1.0], dtype=float))
        initial_z_axis = initial_z_axis / np.linalg.norm(initial_z_axis)

        def project_to_initial_xy(v):
            v = np.asarray(v, dtype=float)
            v = v - np.dot(v, initial_z_axis) * initial_z_axis
            n = np.linalg.norm(v)
            if n < 1e-8:
                return None
            return v / n

        initial_ref = project_to_initial_xy(
            initial_rot.apply(np.array([1.0, 0.0, 0.0], dtype=float))
        )
        current_ref = project_to_initial_xy(
            current_rot.apply(np.array([1.0, 0.0, 0.0], dtype=float))
        )

        # Fallback is only for degenerate poses where +X is almost parallel to
        # the initial z axis. It should not happen for a normal screw pose, but
        # keeps the success check from crashing on bad simulation states.
        if initial_ref is None or current_ref is None:
            initial_ref = project_to_initial_xy(
                initial_rot.apply(np.array([0.0, 1.0, 0.0], dtype=float))
            )
            current_ref = project_to_initial_xy(
                current_rot.apply(np.array([0.0, 1.0, 0.0], dtype=float))
            )

        if initial_ref is None or current_ref is None:
            return 0.0

        sin_angle = np.dot(np.cross(initial_ref, current_ref), initial_z_axis)
        cos_angle = np.dot(initial_ref, current_ref)
        return float(np.arctan2(sin_angle, cos_angle))

    def is_success(self, rotation_threshold_deg=33.0, require_contact=True):
        """Success = wrench tool contacts screw and screw rotates >= threshold.

        The rotation threshold is measured around the screw's initial local z
        axis. By default, success requires at least 40 degrees of absolute
        rotation from the settled initial screw orientation saved in make_scene().
        """
        if not hasattr(self, "screw_obj_id"):
            return False

        if not hasattr(self, "wrench_body_id") or self.wrench_body_id is None:
            return False

        if self.initial_obj_orn is None:
            return False

        if require_contact:
            contacts = self.bullet_client.getContactPoints(
                bodyA=self.wrench_body_id,
                bodyB=self.screw_obj_id,
            )
            wrench_contact_screw = len(contacts) > 0
        else:
            wrench_contact_screw = True

        screw_angle_rad = abs(self.get_screw_rotation_about_initial_z())
        rotated_enough = screw_angle_rad >= math.radians(float(rotation_threshold_deg))

        return bool(wrench_contact_screw and rotated_enough)

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
