import time
import cv2
import gym
import numpy as np
import math
import pybullet_data
from VisualDR import (
    LightingDR, DistractorDR, PoseDR, ObjectColorDR,
    ImgNoiseDR, IntrinsicDR, TextureDR,
)
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
    -0.09762531509002051,
    -0.11921289903971187,
    -0.0004173343487966217,
    -2.1853378428408976,
    -0.0002942036915626864,
    2.064449508160775,
    0.687251996585265,
    0.04,
    0.04
]
rp = jointPositions


class AssemblySim(object):
    def __init__(self, bullet_client, offset, 
                 control_dt = 1.0 / 120.0, seed = 42,
                 randomize_initial_ee_pose = True,
                 initial_ee_x_jit = 0.04,
                 initial_ee_y_jit = 0.05,
                 initial_ee_z_jit = 0.05,
                 initial_ee_eul_jit = 0.12):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

        # Visual domain randomizers. Each randomizer owns its own RNG and state.
        # Keep bullet_client inside every module so each multiprocessing worker can
        # create an independent PickUpSim / VisualDR stack.
        self.lightingDR = LightingDR(self.bullet_client, seed=seed)

        # Camera-specific image-level DR. Keep scene-level randomization shared
        # across cameras, but model each camera's own ISP / white balance /
        # exposure / gamma / sensor noise with independent RNG states.
        self.agentviewImgDR = ImgNoiseDR(
            self.bullet_client,
            seed=seed,
            camera_name="agentview",
        )
        self.eyeImgDR = ImgNoiseDR(
            self.bullet_client,
            seed=seed + 1000,
            camera_name="eye_in_hand",
        )
        # Backward-compatible alias for any old external debug code.
        self.ImgNoiseDR = self.agentviewImgDR

        self.parentPoseDR = PoseDR(self.bullet_client, seed=seed)
        self.childPoseDR = PoseDR(self.bullet_client, seed=seed + 10)
        # Backward-compatible alias used by some external debug code.
        self.objposeDR = self.parentPoseDR
        self.camposeDR  = PoseDR(self.bullet_client, seed=seed)
        self.outsceneDR = PoseDR(self.bullet_client, seed=seed)
        self.collplaneDR = PoseDR(self.bullet_client, seed=seed)
        self.distractorDR = DistractorDR(self.bullet_client, seed=seed)
        self.objectColorDR = ObjectColorDR(self.bullet_client, seed=seed)
        self.robotTextureDR = TextureDR(self.bullet_client, seed=seed + 4000)
        self.fisheyeCamDR = PoseDR(self.bullet_client, seed=seed)
        self.initialEePoseDR = PoseDR(self.bullet_client, seed=seed + 2000)
        self.camIntrinsicDR = IntrinsicDR(seed=seed + 3000)
        # Post-grasp object-in-hand pose randomization. The nominal grasp target
        # remains fixed; only the parent pose relative to the EE is perturbed
        # after the gripper closes.
        self.objectInHandPoseDR = PoseDR(self.bullet_client, seed=seed + 5000)
        # Backward-compatible alias for external code that still references it.
        self.initialGraspDR = self.objectInHandPoseDR
        self.object_in_hand_rng = np.random.default_rng(seed + 5000)

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
        self.agentview_intrinsic = self.agentview_base_intrinsic.copy()

        # Eye-in-hand camera calibration: equidistant / Kannala-Brandt 4 (KB4).
        # The real camera is 640x480, but the policy consumes a 224x224 image.
        # Therefore we render a low-res cubemap and directly remap it into the
        # final 224x224 KB4 fisheye observation.
        self.eye_raw_width = 640
        self.eye_raw_height = 480
        self.eye_obs_width = 224
        self.eye_obs_height = 224
        self.eye_face_size = 256  # use 320 for a little sharper image, 512 for debugging
        self.eye_near = 0.005
        self.eye_far = 1.0

        self.eye_base_K = np.array([
            [242.78325327,   0.0,        316.28355305],
            [  0.0,        243.37493553, 211.80725024],
            [  0.0,          0.0,          1.0],
        ], dtype=np.float64)
        self.eye_K = self.eye_base_K.copy()

        self.eye_D = np.array(
            [-0.04749038, 0.01991722, -0.02552236, 0.0084535],
            dtype=np.float64,
        )

        # Calibration convention assumed here:
        #   T_world_cam = T_world_eye_parent @ T_eye_parent_cam
        # User-provided physical meaning: camera is mounted on the Franka
        # panda_joint7 base coordinate, about +5.1 cm forward and -7.3 cm down.
        self.T_eye_base_parent_cam = np.array([
            [0, -1, 0,  0.05054945],
            [1,  0, 0, -0.00619893],
            [0,  0, 1,  0.01294445],
            [0,  0, 0,  1.0],
        ], dtype=np.float64)

        self.eye_parent_link = 8
    
        # Five 90-degree faces cover this KB4 camera. The farthest rays are just
        # beyond 90 degrees from optical axis, so the back face is unnecessary.
        self.eye_face_names = ("pos_z", "pos_x", "neg_x", "pos_y", "neg_y")
        self.eye_face_basis = self._build_eye_face_basis()
        self.eye_face_proj_90 = self.bullet_client.computeProjectionMatrixFOV(
            fov=90.0,
            aspect=1.0,
            nearVal=self.eye_near,
            farVal=self.eye_far,
        )
        self.eye_fisheye_remap = self.build_eye_fisheye_remap(
            out_width=self.eye_obs_width,
            out_height=self.eye_obs_height,
            face_size=self.eye_face_size,
        )

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

        self.safe_approach = 0.05
        self.safe_grasp_offset = 0.05
        self.arm_force = 200.0
        self.gripper_force = 200.0

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
        # Home pose and joints. Optionally randomize around the initial gripper EE pose
        # before caching home, so the whole state machine starts from this sampled pose.
        base_ee_pos, base_ee_orn = self.get_ee_pose()
        if randomize_initial_ee_pose:
            init_ee_pos, init_ee_orn = self.initialEePoseDR.sample_SE3_randomization(
                pos=base_ee_pos,
                orn=base_ee_orn,
                x_jitter_range=initial_ee_x_jit,
                y_jitter_range=initial_ee_y_jit,
                z_jitter_range=initial_ee_z_jit,
                x_euler_jitter_range=initial_ee_eul_jit,
                y_euler_jitter_range=initial_ee_eul_jit,
                z_euler_jitter_range=initial_ee_eul_jit,
            )
            self.solve_ik_and_apply(init_ee_pos, init_ee_orn, input_frame="parent_ee", reset=True)

        self.home_joint = np.array(self.get_current_arm_joints(), dtype=float)
        self.home_ee_pos, self.home_ee_orn = self.get_ee_pose()
        self.home_ee_pos = np.array(self.home_ee_pos, dtype=float)
        self.home_ee_orn = np.array(self.home_ee_orn, dtype=float)

        #########========= assembly state machine =========###################
        self.states = [
            "home",
            "move_pregrasp",
            "open_gripper",
            "move_grasp",
            "close_gripper",
            "lift_parent",
            "move_preassembly",
            "assemble",
            "hold_assembled",
        ]
        self.state_durations = [0.01, 4.0, 0.25, 2.0, 0.75, 2.0, 4.0, 3.0, 0.5]
        self.state_idx = 0
        self.state = self.states[self.state_idx]
        self.state_t = 0.0

        # target pose, target gripper state (1=open, 0=closed)
        self.target_pos = None
        self.target_orn = None
        self.target_gripper = None
        self.done = False

        # Current state's interpolated motion segment.
        self.motion_start_pos = None
        self.motion_start_orn = None
        self.motion_target_pos = None
        self.motion_target_orn = None

        # Grasp and assembly references.
        self.last_grasp_pose = None
        self.last_grasp_orn = None
        self.parent_motion_start_pos = None
        self.parent_motion_start_orn = None
        self.parent_motion_target_pos = None
        self.parent_motion_target_orn = None
        self.initial_parent_pos = None
        self.initial_parent_orn = None
        self.initial_child_pos = None
        self.initial_child_orn = None

        # Post-grasp rigid attachment / object-in-hand DR state.
        self.parent_grasp_constraint_id = None
        self.grasp_parent_to_ee = None
        self.nominal_ee_to_parent = None
        self.randomized_ee_to_parent = None
        self.object_in_hand_delta_pos = np.zeros(3, dtype=float)
        self.object_in_hand_delta_euler = np.zeros(3, dtype=float)

        # Defaults are overwritten by make_scene().
        self.fix_parent_to_gripper = True
        self.randomize_object_in_hand_pose = True
        self.object_in_hand_pos_jit = np.array([0.002, 0.002, 0.003], dtype=float)
        self.object_in_hand_eul_jit = np.array([0.0174533, 0.0174533, 0.0349066], dtype=float)
        self.object_in_hand_debug = False

        self.prepare_state(self.state)

 

    def get_parent_obj_pose(self):
        return tuple(
            np.asarray(v, dtype=float)
            for v in get_true_PositionAndOrientation(
                self.bullet_client, self.assembly_parent_id
            )
        )

    def get_child_obj_pose(self):
        return tuple(
            np.asarray(v, dtype=float)
            for v in get_true_PositionAndOrientation(
                self.bullet_client, self.assembly_child_id
            )
        )

    def get_parent_to_ee_transform(self, ee_pos=None, ee_orn=None):
        """Return the current parent-object-frame -> EE transform.

        With no explicit EE pose, this reads both bodies from PyBullet every
        time, so grasp slip is reflected immediately. An explicit EE pose is
        only used when building annotation-based debug/distractor waypoints.
        """
        parent_pos, parent_orn = self.get_parent_obj_pose()
        if ee_pos is None or ee_orn is None:
            ee_pos, ee_orn = self.get_ee_pose()

        inv_parent_pos, inv_parent_orn = self.bullet_client.invertTransform(
            parent_pos.tolist(), parent_orn.tolist()
        )
        rel_pos, rel_orn = self.bullet_client.multiplyTransforms(
            inv_parent_pos,
            inv_parent_orn,
            np.asarray(ee_pos, dtype=float).tolist(),
            np.asarray(ee_orn, dtype=float).tolist(),
        )
        return np.asarray(rel_pos, dtype=float), np.asarray(rel_orn, dtype=float)

    def remove_parent_grasp_constraint(self):
        """Remove the rigid parent-to-gripper attachment, if one exists."""
        constraint_id = getattr(self, "parent_grasp_constraint_id", None)
        if constraint_id is not None:
            try:
                self.bullet_client.removeConstraint(constraint_id)
            except Exception:
                pass
        self.parent_grasp_constraint_id = None
        self.grasp_parent_to_ee = None

    def get_ee_to_parent_transform(self, ee_pos=None, ee_orn=None):
        """Return the parent-object pose expressed in the EE link frame."""
        parent_pos, parent_orn = self.get_parent_obj_pose()
        if ee_pos is None or ee_orn is None:
            ee_pos, ee_orn = self.get_ee_pose()

        inv_ee_pos, inv_ee_orn = self.bullet_client.invertTransform(
            np.asarray(ee_pos, dtype=float).tolist(),
            np.asarray(ee_orn, dtype=float).tolist(),
        )
        rel_pos, rel_orn = self.bullet_client.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            parent_pos.tolist(),
            parent_orn.tolist(),
        )
        return np.asarray(rel_pos, dtype=float), np.asarray(rel_orn, dtype=float)

    def randomize_and_fix_parent_to_gripper(self):
        """Sample an object-in-hand SE(3) pose and rigidly attach the parent.

        The nominal grasp trajectory is executed first. At the end of
        ``close_gripper`` we measure the actual parent-object pose in the EE
        frame, add a small translation/orientation perturbation in that frame,
        teleport the parent to the sampled pose, and create a fixed constraint.

        No virtual force or physical slip model is used. The purpose is to
        generate image/proprioception-to-assembly-action mappings under varied
        object-in-hand poses.
        """
        if not self.fix_parent_to_gripper:
            self.grasp_parent_to_ee = None
            return

        self.remove_parent_grasp_constraint()

        ee_pos, ee_orn = self.get_ee_pose()
        parent_true_pos, parent_true_orn = self.get_parent_obj_pose()
        parent_base_pos, parent_base_orn = self.bullet_client.getBasePositionAndOrientation(
            self.assembly_parent_id
        )
        parent_base_pos = np.asarray(parent_base_pos, dtype=float)
        parent_base_orn = np.asarray(parent_base_orn, dtype=float)

        # Nominal object pose in EE coordinates, based on the object/mesh frame
        # used by the assembly target calculations.
        nominal_rel_pos, nominal_rel_orn = self.get_ee_to_parent_transform(
            ee_pos=ee_pos, ee_orn=ee_orn
        )
        self.nominal_ee_to_parent = (
            nominal_rel_pos.copy(),
            nominal_rel_orn.copy(),
        )

        if self.randomize_object_in_hand_pose:
            delta_pos = self.object_in_hand_rng.uniform(
                low=-self.object_in_hand_pos_jit,
                high=self.object_in_hand_pos_jit,
            )
            delta_euler = self.object_in_hand_rng.uniform(
                low=-self.object_in_hand_eul_jit,
                high=self.object_in_hand_eul_jit,
            )
        else:
            delta_pos = np.zeros(3, dtype=float)
            delta_euler = np.zeros(3, dtype=float)

        # Translation and rotation are perturbed independently in the EE frame.
        # This avoids rotating the nominal grasp translation around the EE origin.
        randomized_rel_pos = nominal_rel_pos + delta_pos
        randomized_rel_orn = (
            R.from_euler("xyz", delta_euler) * R.from_quat(nominal_rel_orn)
        ).as_quat()
        randomized_rel_orn /= np.linalg.norm(randomized_rel_orn)

        self.object_in_hand_delta_pos = delta_pos.copy()
        self.object_in_hand_delta_euler = delta_euler.copy()
        self.randomized_ee_to_parent = (
            randomized_rel_pos.copy(),
            randomized_rel_orn.copy(),
        )

        # Desired object/mesh-frame pose in world coordinates.
        randomized_true_pos, randomized_true_orn = self.bullet_client.multiplyTransforms(
            ee_pos.tolist(),
            ee_orn.tolist(),
            randomized_rel_pos.tolist(),
            randomized_rel_orn.tolist(),
        )

        # Preserve the fixed transform from the object's true/mesh frame to the
        # PyBullet base frame. resetBasePositionAndOrientation and the fixed
        # constraint operate on the base frame, while assembly targets use the
        # true/mesh frame returned by get_true_PositionAndOrientation().
        inv_true_pos, inv_true_orn = self.bullet_client.invertTransform(
            parent_true_pos.tolist(), parent_true_orn.tolist()
        )
        true_to_base_pos, true_to_base_orn = self.bullet_client.multiplyTransforms(
            inv_true_pos,
            inv_true_orn,
            parent_base_pos.tolist(),
            parent_base_orn.tolist(),
        )
        randomized_base_pos, randomized_base_orn = self.bullet_client.multiplyTransforms(
            randomized_true_pos,
            randomized_true_orn,
            true_to_base_pos,
            true_to_base_orn,
        )

        self.bullet_client.resetBasePositionAndOrientation(
            self.assembly_parent_id,
            randomized_base_pos,
            randomized_base_orn,
        )
        self.bullet_client.resetBaseVelocity(
            self.assembly_parent_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

        # Fixed-constraint parent frame is the EE link; child frame is the
        # PyBullet parent-object base.
        inv_ee_pos, inv_ee_orn = self.bullet_client.invertTransform(
            ee_pos.tolist(), ee_orn.tolist()
        )
        ee_to_base_pos, ee_to_base_orn = self.bullet_client.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            randomized_base_pos,
            randomized_base_orn,
        )
        self.parent_grasp_constraint_id = self.bullet_client.createConstraint(
            parentBodyUniqueId=self.panda,
            parentLinkIndex=pandaEndEffectorIndex,
            childBodyUniqueId=self.assembly_parent_id,
            childLinkIndex=-1,
            jointType=self.bullet_client.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=ee_to_base_pos,
            childFramePosition=[0.0, 0.0, 0.0],
            parentFrameOrientation=ee_to_base_orn,
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
        )

        # Cache parent-object -> EE for all subsequent expert targets.
        cached_pos, cached_orn = self.bullet_client.invertTransform(
            randomized_rel_pos.tolist(), randomized_rel_orn.tolist()
        )
        self.grasp_parent_to_ee = (
            np.asarray(cached_pos, dtype=float),
            np.asarray(cached_orn, dtype=float),
        )

        if self.object_in_hand_debug:
            print(
                "object-in-hand DR:",
                "delta_pos_ee=", np.round(delta_pos, 6),
                "delta_euler_deg=", np.round(np.degrees(delta_euler), 3),
                "constraint_id=", self.parent_grasp_constraint_id,
            )

    def get_ee_pose_for_parent_pose(
        self, parent_pos, parent_orn, parent_to_ee=None
    ):
        """Convert a desired parent-object pose into an EE pose.

        Runtime assembly calls leave ``parent_to_ee`` unset and re-measure the
        transform every step. After rigid attachment this transform should be
        constant apart from tiny solver noise, matching the continuously
        measured parent-to-EE mapping used by the scripted expert.
        """
        if parent_to_ee is None:
            parent_to_ee = self.get_parent_to_ee_transform()
        rel_pos, rel_orn = parent_to_ee
        ee_pos, ee_orn = self.bullet_client.multiplyTransforms(
            np.asarray(parent_pos, dtype=float).tolist(),
            np.asarray(parent_orn, dtype=float).tolist(),
            np.asarray(rel_pos, dtype=float).tolist(),
            np.asarray(rel_orn, dtype=float).tolist(),
        )
        return np.asarray(ee_pos, dtype=float), np.asarray(ee_orn, dtype=float)

    def get_assembly_target_parent_pose(self):
        """Desired world pose of the square ring at the final assembly location.

        The parent OBJ origin is the ring center. Therefore the default target is
        exactly the child OBJ origin, with an optional child-frame offset.
        """
        child_pos, child_orn = self.get_child_obj_pose()
        target_pos, target_orn = self.bullet_client.multiplyTransforms(
            child_pos.tolist(),
            child_orn.tolist(),
            self.child_to_parent_target_pos.tolist(),
            self.child_to_parent_target_orn.tolist(),
        )
        return np.asarray(target_pos, dtype=float), np.asarray(target_orn, dtype=float)

    def get_preassembly_parent_pose(self):
        """Return a pose above the final target along the child local assembly axis."""
        target_pos, target_orn = self.get_assembly_target_parent_pose()
        _, child_orn = self.get_child_obj_pose()
        child_rot = R.from_quat(child_orn)
        axis_world = child_rot.apply(self.assembly_axis_local)
        axis_world = axis_world / np.linalg.norm(axis_world)
        pre_pos = target_pos + self.safe_assembly_approach * axis_world
        return pre_pos, target_orn

    def make_scene(
                   self,
                   env_mesh_path=None,
                   assembly_parent_path=None,
                   assembly_parent_collision_path=None,
                   assembly_child_path=None,
                   assembly_child_collision_path=None,
                   initial_grasp_path=None,
                   # Kept only so the existing demo remains call-compatible.
                   # FPSA is intentionally disabled for AssemblySim.
                   if_FPSA_tool=False,
                   fpsa_tool_aug_root=None,
                   fpsa_tool_include_base=False,
                   wrench_collision_path=None,
                   parentobj_pose_base=(0.45, -0.05, 0.05),
                   parentobj_euler_base=(0.0, 0.0, 0.0),
                   childobj_pose_base=(0.70, -0.05, 0.05),
                   childobj_euler_base=(0.0, 0.0, 0.0),
                   randomize_lighting=True,
                   randomize_outlscene=True,
                   outlscene_xyz_jit=0.02,
                   outlscene_eul_jit=0.01,
                   randomize_plane_height=True,
                   plane_height_jit=0.008,
                   randomize_parent_objpose=True,
                   parentobj_x_jit=0.05,
                   parentobj_y_jit=0.10,
                   parentobj_z_jit=0.05,
                   parentobj_z_eul_jit=0.0,
                   randomize_child_objpose=True,
                   childobj_x_jit=0.05,
                   childobj_y_jit=0.10,
                   childobj_z_jit=0.05,
                   childobj_z_eul_jit=0.0,
                   randomize_campose=True,
                   cam_xyz_jit=0.004,
                   cam_eul_jit=0.002,
                   randomize_fisheye_cam=True,
                   fisheye_eyz_jit=0.005,
                   fisheye_eul_jit=0.002,
                   randomize_camera_intrinsic=True,
                   agentview_focal_scale_range=(0.88, 1.15),
                   agentview_principal_jit_px=18.0,
                   eye_focal_scale_range=(0.90, 1.12),
                   eye_principal_jit_px=8.0,
                   randomize_image_noise=True,
                   randomize_robot_texture=True,
                   robot_texture_patterns=("checkers", "gradient", "noise", "plain"),
                   robot_texture_size=128,
                   robot_texture_per_link=True,
                   robot_texture_specular_range=(0.02, 0.25),
                   robot_original_texture_prob=0.10,
                   randomize_object_color=True,
                   object_color_mode="bounded",
                   object_color_strength=0.35,
                   object_recolor_palette=None,
                   object_recolor_target_color=None,
                   object_specular_range=(0.02, 0.5),
                   # Legacy wrench color arguments are accepted but intentionally ignored.
                   randomize_wrench_color=False,
                   wrench_color_mode="bounded",
                   wrench_color_strength=0.35,
                   randomize_distractors=True,
                   distractor_root="/mnt/storage/GoogleScannedObjects",
                   distractor_num_range=(1, 5),
                   distractor_target_size_range=(0.06, 0.16),
                   distractor_workspace=((0.25, 0.78), (-0.42, 0.42)),
                   distractor_clearance=0.04,
                   distractor_path_clearance=0.04,
                   distractor_min_target_mask_pixels=1,
                   parent_mass=0.85,
                   child_mass=0.2,
                   parent_lateral_friction=0.9,
                   child_lateral_friction=0.8,
                   child_rgba=(0.90, 0.03, 0.03, 1.0),
                   assembly_axis_local=(0.0, 0.0, 1.0),
                   child_to_parent_target_pos=(0.0, 0.0, 0.0),
                   child_to_parent_target_euler=(0.0, 0.0, 0.0),
                   safe_assembly_approach=0.10,
                   # Post-grasp object-in-hand domain randomization. Nominal
                   # pregrasp/grasp targets are never randomized.
                   fix_parent_to_gripper=True,
                   randomize_object_in_hand_pose=True,
                   object_in_hand_x_jit=0.002,
                   object_in_hand_y_jit=0.002,
                   object_in_hand_z_jit=0.003,
                   object_in_hand_roll_jit=0.0174533,
                   object_in_hand_pitch_jit=0.0174533,
                   object_in_hand_yaw_jit=0.0349066,
                   object_in_hand_debug=False,
                   ):
        """Build a pick-and-assemble scene with a movable parent ring and fixed child rod."""
        del fpsa_tool_aug_root, fpsa_tool_include_base, wrench_collision_path
        del randomize_wrench_color, wrench_color_mode, wrench_color_strength
        # Explicitly hard-disable FPSA as requested, even if an old caller passes True.
        self.if_FPSA_tool = False
        if if_FPSA_tool:
            print("AssemblySim: if_FPSA_tool=True was ignored; FPSA is hard-disabled.")

        if assembly_parent_path is None:
            raise ValueError("assembly_parent_path must be provided")
        if assembly_child_path is None:
            raise ValueError("assembly_child_path must be provided")
        if initial_grasp_path is None:
            raise ValueError("initial_grasp_path must be provided")

        self.env_mesh_path = env_mesh_path
        self.assembly_parent_path = assembly_parent_path
        self.assembly_child_path = assembly_child_path
        self.initial_grasp_path = initial_grasp_path
        self.safe_assembly_approach = float(safe_assembly_approach)

        # Clear any attachment left by a previous scene build, then configure
        # the post-grasp object-in-hand randomization for this episode.
        self.remove_parent_grasp_constraint()
        self.fix_parent_to_gripper = bool(fix_parent_to_gripper)
        self.randomize_object_in_hand_pose = bool(randomize_object_in_hand_pose)
        self.object_in_hand_pos_jit = np.array(
            [object_in_hand_x_jit, object_in_hand_y_jit, object_in_hand_z_jit],
            dtype=float,
        )
        self.object_in_hand_eul_jit = np.array(
            [object_in_hand_roll_jit, object_in_hand_pitch_jit, object_in_hand_yaw_jit],
            dtype=float,
        )
        if np.any(self.object_in_hand_pos_jit < 0.0):
            raise ValueError("object-in-hand position jitter ranges must be non-negative")
        if np.any(self.object_in_hand_eul_jit < 0.0):
            raise ValueError("object-in-hand Euler jitter ranges must be non-negative")
        self.object_in_hand_debug = bool(object_in_hand_debug)
        self.grasp_parent_to_ee = None
        self.nominal_ee_to_parent = None
        self.randomized_ee_to_parent = None

        self.assembly_axis_local = np.asarray(assembly_axis_local, dtype=float)
        axis_norm = np.linalg.norm(self.assembly_axis_local)
        if axis_norm < 1e-8:
            raise ValueError("assembly_axis_local must be non-zero")
        self.assembly_axis_local /= axis_norm
        self.child_to_parent_target_pos = np.asarray(
            child_to_parent_target_pos, dtype=float
        )
        self.child_to_parent_target_orn = np.asarray(
            self.bullet_client.getQuaternionFromEuler(
                np.asarray(child_to_parent_target_euler, dtype=float)
            ),
            dtype=float,
        )

        parentobj_pose_base = np.asarray(parentobj_pose_base, dtype=float).copy()
        parentobj_euler_base = np.asarray(parentobj_euler_base, dtype=float).copy()
        childobj_pose_base = np.asarray(childobj_pose_base, dtype=float).copy()
        childobj_euler_base = np.asarray(childobj_euler_base, dtype=float).copy()

        self.parent_collision_path = (
            assembly_parent_collision_path
            if assembly_parent_collision_path is not None
            else coacd_convex_decomposition(self.assembly_parent_path)
        )
        self.child_collision_path = (
            assembly_child_collision_path
            if assembly_child_collision_path is not None
            else coacd_convex_decomposition(self.assembly_child_path)
        )
        self.com_parent = get_com(self.assembly_parent_path)
        self.com_child = get_com(self.assembly_child_path)
        self.initial_grasp_guess = load_initial_grasp_pose(initial_grasp_path)

        if randomize_lighting:
            self.lightingDR.sample_lighting_randomization()
        else:
            self.lightingDR.reset_to_default()

        if randomize_robot_texture:
            self.robot_texture_cfg = self.robotTextureDR.sample_and_apply_robot_texture_randomization(
                body_id=self.panda,
                patterns=robot_texture_patterns,
                texture_size=robot_texture_size,
                per_link=robot_texture_per_link,
                specular_range=robot_texture_specular_range,
                alpha=None,
                original_texture_prob=robot_original_texture_prob,
            )
            # ic(self.robot_texture_cfg)
        else:
            self.robotTextureDR.reset(body_id=self.panda, restore_original=True)

        parent_base_orn = self.bullet_client.getQuaternionFromEuler(parentobj_euler_base)
        if randomize_parent_objpose:
            parent_pose, parent_orn = self.parentPoseDR.sample_SE3_randomization(
                pos=parentobj_pose_base,
                orn=parent_base_orn,
                x_jitter_range=parentobj_x_jit,
                y_jitter_range=parentobj_y_jit,
                z_jitter_range=parentobj_z_jit,
                z_euler_jitter_range=parentobj_z_eul_jit,
            )
        else:
            parent_pose, parent_orn = parentobj_pose_base, parent_base_orn

        child_base_orn = self.bullet_client.getQuaternionFromEuler(childobj_euler_base)
        if randomize_child_objpose:
            child_pose, child_orn = self.childPoseDR.sample_SE3_randomization(
                pos=childobj_pose_base,
                orn=child_base_orn,
                x_jitter_range=childobj_x_jit,
                y_jitter_range=childobj_y_jit,
                z_jitter_range=childobj_z_jit,
                z_euler_jitter_range=childobj_z_eul_jit,
            )
        else:
            child_pose, child_orn = childobj_pose_base, child_base_orn

        if randomize_image_noise:
            self.agentviewImgDR.sample_image_noise_randomization(
                brightness_range=(-16.0, 16.0),
                contrast_range=(0.85, 1.18),
                gamma_range=(0.85, 1.20),
                saturation_range=(0.75, 1.30),
                rgb_gain_range=(0.85, 1.15),
                hue_shift_deg_range=(-5.0, 5.0),
                color_matrix_strength_range=(0.0, 0.08),
                gray_mix_range=(0.0, 0.12),
                vignette_strength_range=(0.0, 0.10),
                gaussian_std_range=(0.0, 3.0),
                salt_pepper_prob_range=(0.0, 0.0015),
                blur_prob_range=(0.0, 0.12),
            )
            self.eyeImgDR.sample_image_noise_randomization(
                brightness_range=(-20.0, 20.0),
                contrast_range=(0.82, 1.22),
                gamma_range=(0.82, 1.25),
                saturation_range=(0.70, 1.35),
                rgb_gain_range=(0.82, 1.18),
                hue_shift_deg_range=(-6.0, 6.0),
                color_matrix_strength_range=(0.0, 0.10),
                gray_mix_range=(0.0, 0.16),
                vignette_strength_range=(0.0, 0.16),
                gaussian_std_range=(0.0, 3.5),
                salt_pepper_prob_range=(0.0, 0.002),
                blur_prob_range=(0.0, 0.16),
            )
        else:
            self.agentviewImgDR.reset()
            self.eyeImgDR.reset()

        outscene_base_pos = np.array([0.0, 0.0, -0.005], dtype=float)
        outscene_base_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if randomize_outlscene:
            env_mesh_pos, env_mesh_orn = self.outsceneDR.sample_SE3_randomization(
                pos=outscene_base_pos,
                orn=outscene_base_orn,
                x_jitter_range=outlscene_xyz_jit,
                y_jitter_range=outlscene_xyz_jit,
                z_jitter_range=outlscene_xyz_jit,
                x_euler_jitter_range=None,
                y_euler_jitter_range=None,
                z_euler_jitter_range=outlscene_eul_jit,
            )
        else:
            env_mesh_pos, env_mesh_orn = outscene_base_pos, outscene_base_orn

        if randomize_plane_height:
            ground_pose = self.collplaneDR.sample_pos_randomization(
                pos=[0.0, 0.0, env_mesh_pos[2]],
                z_jitter_range=plane_height_jit,
            )
        else:
            ground_pose = np.array([0.0, 0.0, env_mesh_pos[2]], dtype=np.float32)

        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_plane_id = self.bullet_client.loadURDF(
            "plane.urdf", basePosition=ground_pose
        )
        self.bullet_client.changeVisualShape(
            self.ground_plane_id, -1, rgbaColor=[1, 1, 1, 0]
        )

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

        if randomize_fisheye_cam:
            self.T_eye_parent_cam = self.fisheyeCamDR.sample_SE3_randomization(
                pos=self.T_eye_base_parent_cam[:3, 3],
                orn=quat_from_rotation_matrix(self.T_eye_base_parent_cam[:3, :3]),
                x_jitter_range=fisheye_eyz_jit,
                y_jitter_range=fisheye_eyz_jit,
                z_jitter_range=fisheye_eyz_jit,
                x_euler_jitter_range=fisheye_eul_jit,
                y_euler_jitter_range=fisheye_eul_jit,
                z_euler_jitter_range=fisheye_eul_jit,
                get_matrix=True,
            )
        else:
            self.T_eye_parent_cam = self.T_eye_base_parent_cam.copy()

        if randomize_camera_intrinsic:
            self.agentview_intrinsic = self.camIntrinsicDR.sample_intrinsic_randomization(
                self.agentview_base_intrinsic,
                focal_scale_range=agentview_focal_scale_range,
                principal_jit_px=agentview_principal_jit_px,
                width=self.agentview_width,
                height=self.agentview_height,
            )
            self.eye_K = self.camIntrinsicDR.sample_intrinsic_randomization(
                self.eye_base_K,
                focal_scale_range=eye_focal_scale_range,
                principal_jit_px=eye_principal_jit_px,
                width=self.eye_raw_width,
                height=self.eye_raw_height,
            )
        else:
            self.agentview_intrinsic = self.agentview_base_intrinsic.copy()
            self.eye_K = self.eye_base_K.copy()

        self.eye_fisheye_remap = self.build_eye_fisheye_remap(
            out_width=self.eye_obs_width,
            out_height=self.eye_obs_height,
            face_size=self.eye_face_size,
        )

        self.env_mesh = load_models(
            self.bullet_client,
            visual_mesh_file=self.env_mesh_path,
            vhacd_mesh_file=None,
            desired_mass=0.0,
            position=env_mesh_pos,
            baseOrientation=env_mesh_orn,
            visual_only=True,
        )

        self.assembly_child_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.assembly_child_path,
            vhacd_mesh_file=self.child_collision_path,
            desired_mass=float(child_mass),
            position=child_pose,
            baseOrientation=child_orn,
            center_of_mass=np.asarray(self.com_child),
            lateral_friction=float(child_lateral_friction),
            spinning_friction=0.0,
        )
        self.child_obj_id = self.assembly_child_id
        for link_idx in range(-1, self.bullet_client.getNumJoints(self.assembly_child_id)):
            try:
                self.bullet_client.changeVisualShape(
                    self.assembly_child_id,
                    link_idx,
                    rgbaColor=list(child_rgba),
                )
            except Exception:
                pass

        self.assembly_parent_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.assembly_parent_path,
            vhacd_mesh_file=self.parent_collision_path,
            desired_mass=float(parent_mass),
            position=parent_pose,
            baseOrientation=parent_orn,
            center_of_mass=np.asarray(self.com_parent),
            lateral_friction=float(parent_lateral_friction),
            spinning_friction=0.0002,
        )
        self.parent_obj_id = self.assembly_parent_id
        self.pick_up_obj_id = self.assembly_parent_id

        # Improve physical grasp stability without rigidly attaching the object.
        for finger_link in (9, 10):
            self.bullet_client.changeDynamics(
                self.panda,
                finger_link,
                lateralFriction=1.2,
                spinningFriction=0.01,
                rollingFriction=0.001,
            )

        if randomize_object_color:
            self.object_color_cfg = self.objectColorDR.sample_and_apply_object_color_randomization(
                body_id=self.assembly_parent_id,
                mode=object_color_mode,
                strength=object_color_strength,
                recolor_palette=object_recolor_palette,
                recolor_target_color=object_recolor_target_color,
                specular_range=object_specular_range,
                alpha=None,)
            
            self.object_color_cfg = self.objectColorDR.sample_and_apply_object_color_randomization(
                body_id=self.assembly_child_id,
                mode=object_color_mode,
                strength=object_color_strength,
                recolor_palette=object_recolor_palette,
                recolor_target_color=object_recolor_target_color,
                specular_range=object_specular_range,
                alpha=None,
            )
        else:
            self.objectColorDR.reset()
            self.object_color_cfg = None

        self.waite_scene_stable()

        if randomize_distractors:
            self.distractorDR.sample_and_load_distractors(
                distractor_root=distractor_root,
                num_range=distractor_num_range,
                target_size_range=distractor_target_size_range,
                workspace=distractor_workspace,
                clearance=distractor_clearance,
                path_clearance=distractor_path_clearance,
                min_target_mask_pixels=distractor_min_target_mask_pixels,
                target_body_id=self.assembly_parent_id,
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
                ground_z=float(ground_pose[2]),
                spawn_clearance=0.005,
                check_robot_plan=True,
                check_xy_safety=True,
                min_visible_fraction=0.55,
                debug=False,
            )
        else:
            self.distractorDR.clear_distractors()

        for _ in range(5):
            self.bullet_client.stepSimulation()

        self.initial_parent_pos, self.initial_parent_orn = self.get_parent_obj_pose()
        self.initial_child_pos, self.initial_child_orn = self.get_child_obj_pose()
        self.target_gripper = self.GRIPPER_OPEN
        self.set_gripper_state(self.target_gripper)

        # Rebuild the first state now that task assets and grasp annotation exist.
        if self.state == "home":
            self.prepare_state(self.state)

    def get_state_machine_ee_waypoints(self):
        """Approximate the complete pick-up and insertion path for distractor checks."""
        grasp_pos, grasp_orn = self.get_initial_guess_grasp()
        pregrasp_pos = grasp_pos + np.array([0.0, 0.0, self.safe_approach])
        lifted_pos = grasp_pos + np.array([0.0, 0.0, self.safe_grasp_offset + 0.12])

        pre_parent_pos, pre_parent_orn = self.get_preassembly_parent_pose()
        target_parent_pos, target_parent_orn = self.get_assembly_target_parent_pose()
        annotated_parent_to_ee = self.get_parent_to_ee_transform(
            ee_pos=grasp_pos, ee_orn=grasp_orn
        )
        pre_ee_pos, pre_ee_orn = self.get_ee_pose_for_parent_pose(
            pre_parent_pos, pre_parent_orn, annotated_parent_to_ee
        )
        target_ee_pos, target_ee_orn = self.get_ee_pose_for_parent_pose(
            target_parent_pos, target_parent_orn, annotated_parent_to_ee
        )

        return [
            (self.home_ee_pos.copy(), self.home_ee_orn.copy()),
            (pregrasp_pos, grasp_orn.copy()),
            (grasp_pos.copy(), grasp_orn.copy()),
            (lifted_pos, grasp_orn.copy()),
            (pre_ee_pos, pre_ee_orn),
            (target_ee_pos, target_ee_orn),
        ]

    def waite_scene_stable(self, waite_steps=1000, vel_threshold=0.005):
        steps = 0
        while steps < waite_steps:
            self.bullet_client.stepSimulation()
            steps += 1
            vel, ang_vel = self.bullet_client.getBaseVelocity(self.assembly_parent_id)
            speed = np.linalg.norm(vel) + np.linalg.norm(ang_vel)
            if speed < vel_threshold:
                print("Scene stabilized.")
                return True
        print("Warning: parent object did not stabilize within timeout.")
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
        parent_world_pos, parent_world_orn = self.get_parent_obj_pose()
        grasp_pos, raw_grasp_orn = self.bullet_client.multiplyTransforms(
            parent_world_pos.tolist(),
            parent_world_orn.tolist(),
            self.initial_grasp_guess["t"],
            self.initial_grasp_guess["quat"],
        )
        grasp_orn = self.get_fixed_normal_grasp_orn(raw_grasp_orn)
        return np.asarray(grasp_pos, dtype=float), np.asarray(grasp_orn, dtype=float)

    def get_ee_pose(self):
        link_state = self.bullet_client.getLinkState(
            self.panda, pandaEndEffectorIndex, computeForwardKinematics=True
        )
        return np.asarray(link_state[4], dtype=float), np.asarray(link_state[5], dtype=float)

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

    def solve_ik_and_apply(self, target_pos, target_orn, input_frame="ee", reset=False):
        """Solve IK directly for the original Franka Panda end-effector frame."""
        del input_frame
        current_q = self.get_current_arm_joints()
        jointPoses = self.bullet_client.calculateInverseKinematics(
            self.panda,
            pandaEndEffectorIndex,
            np.asarray(target_pos, dtype=float).tolist(),
            np.asarray(target_orn, dtype=float).tolist(),
            ll,
            ul,
            jr,
            current_q,
            maxNumIterations=50,
        )
        for i in range(pandaNumDofs):
            if reset:
                self.bullet_client.resetJointState(self.panda, i, jointPoses[i])
            self.bullet_client.setJointMotorControl2(
                self.panda,
                i,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=jointPoses[i],
                force=self.arm_force,
            )
        return np.asarray(jointPoses[:pandaNumDofs], dtype=float)

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

    def prepare_state(self, state):
        ee_pos, ee_orn = self.get_ee_pose()
        self.motion_start_pos = ee_pos.copy()
        self.motion_start_orn = ee_orn.copy()
        self.motion_target_pos = ee_pos.copy()
        self.motion_target_orn = ee_orn.copy()
        self.parent_motion_start_pos = None
        self.parent_motion_start_orn = None
        self.parent_motion_target_pos = None
        self.parent_motion_target_orn = None

        if self.target_gripper is None:
            self.target_gripper = self.GRIPPER_OPEN

        if state == "home":
            self.motion_target_pos = self.home_ee_pos.copy()
            self.motion_target_orn = self.home_ee_orn.copy()
            self.target_gripper = self.GRIPPER_OPEN
            self.set_gripper_state(self.target_gripper)

        elif state == "move_pregrasp":
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()
            self.motion_target_pos = grasp_pos + np.array(
                [0.0, 0.0, self.safe_approach], dtype=float
            )
            self.motion_target_orn = grasp_orn.copy()

        elif state == "open_gripper":
            self.target_gripper = self.GRIPPER_OPEN

        elif state == "move_grasp":
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()
            self.last_grasp_pose = grasp_pos.copy()
            self.last_grasp_orn = grasp_orn.copy()
            self.motion_target_pos = grasp_pos.copy()
            self.motion_target_orn = grasp_orn.copy()

        elif state == "close_gripper":
            self.target_gripper = self.GRIPPER_CLOSED

        elif state == "lift_parent":
            self.motion_target_pos = ee_pos + np.array(
                [0.0, 0.0, self.safe_grasp_offset + 0.12], dtype=float
            )
            self.motion_target_orn = ee_orn.copy()

        elif state in ("move_preassembly", "assemble"):
            self.parent_motion_start_pos, self.parent_motion_start_orn = (
                self.get_parent_obj_pose()
            )
            if state == "move_preassembly":
                target_parent_pose = self.get_preassembly_parent_pose()
            else:
                target_parent_pose = self.get_assembly_target_parent_pose()
            (
                self.parent_motion_target_pos,
                self.parent_motion_target_orn,
            ) = target_parent_pose

        elif state == "hold_assembled":
            self.motion_target_pos = ee_pos.copy()
            self.motion_target_orn = ee_orn.copy()
            self.target_gripper = self.GRIPPER_CLOSED

    def switch_to_next_state(self):
        # Execute nominal grasp first. Once close_gripper completes, perturb the
        # measured object-in-hand pose and rigidly attach the parent before lift.
        # Data collection can remain continuous; this intentionally models a
        # sampled post-grasp state rather than physical slip dynamics.
        if self.state == "close_gripper":
            self.randomize_and_fix_parent_to_gripper()

        self.state_idx += 1
        if self.state_idx >= len(self.states):
            self.done = True
            return

        self.state = self.states[self.state_idx]
        self.state_t = 0.0
        self.prepare_state(self.state)
        print("state ->", self.state)

    def step(self):
        if self.done:
            return self.target_pos, self.target_orn

        self.t += self.control_dt
        self.state_t += self.control_dt
        duration = self.state_durations[self.state_idx]
        s = min(self.state_t / max(duration, 1e-8), 1.0)

        if self.state in ["open_gripper", "close_gripper"]:
            self.set_gripper_state(self.target_gripper)
            self.target_pos, self.target_orn = self.get_ee_pose()
        elif self.state in ("move_preassembly", "assemble"):
            desired_parent_pos = (
                (1.0 - s) * self.parent_motion_start_pos
                + s * self.parent_motion_target_pos
            )
            desired_parent_orn = quat_slerp(
                self.parent_motion_start_orn,
                self.parent_motion_target_orn,
                s,
            )
            # Continuously measure parent-object -> EE. With the rigid
            # post-grasp constraint this is effectively fixed, while the expert
            # still follows the same live transform-composition path.
            self.target_pos, self.target_orn = self.get_ee_pose_for_parent_pose(
                desired_parent_pos, desired_parent_orn
            )
            self.solve_ik_and_apply(self.target_pos, self.target_orn)
            self.set_gripper_state(self.target_gripper)
        else:
            self.target_pos = (
                (1.0 - s) * self.motion_start_pos + s * self.motion_target_pos
            )
            self.target_orn = quat_slerp(
                self.motion_start_orn, self.motion_target_orn, s
            )
            self.solve_ik_and_apply(self.target_pos, self.target_orn)
            self.set_gripper_state(self.target_gripper)

        if self.state_t >= duration:
            self.switch_to_next_state()

        return self.target_pos, self.target_orn

    def collect_observation(self, use_agent_cam=True, direct=False,
                            collect_wrench_ee=False, use_eye_in_hand=True):
        """Collect policy observations in the original Franka EE frame."""
        del collect_wrench_ee
        eef_pos, eef_quat = self.get_ee_pose()
        finger_widths = np.array([
            self.bullet_client.getJointState(self.panda, 9)[0],
            self.bullet_client.getJointState(self.panda, 10)[0],
        ], dtype=np.float32)
        gripper_state = np.array([
            self.gripper_width_to_state(np.mean(finger_widths))
        ], dtype=np.float32)

        obs = {
            "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32),
            "robot0_eef_quat": np.asarray(eef_quat, dtype=np.float32),
            "robot0_gripper_qpos": gripper_state,
        }

        if use_agent_cam:
            if direct:
                agentview224 = self.direct_get_agent_view()
            else:
                _, _, agentview_rgba, _, _ = self.get_agentview_image()
                agentview224 = resize_rgb(agentview_rgba[..., :3], out_size=224)
            obs["agentview_image"] = agentview224.astype(np.uint8)

        if use_eye_in_hand:
            obs["robot0_eye_in_hand_image"] = self.get_eye_in_hand_image().astype(np.uint8)
        return obs

    def collect_action(self):
        """Return [EE xyz, EE quaternion, binary gripper] with no wrench TCP conversion."""
        gripper = np.array([self.target_gripper], dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(self.target_pos, dtype=float),
                np.asarray(self.target_orn, dtype=float),
                gripper,
            ],
            axis=0,
        ).astype(np.float32)

    def _build_eye_face_basis(self):
        """Return cubemap face frames in the OpenCV camera convention.

        Camera convention used by cvPose2BulletView and the KB4 ray model:
            +X right, +Y down, +Z forward.

        Each 3x3 matrix R_cam_face has columns equal to the face camera local
        x/y/z axes expressed in the original fisheye camera frame. Therefore:
            ray_face = ray_cam @ R_cam_face
        for row-vector rays.
        """
        return {
            # front face: face frame equals fisheye camera frame
            "pos_z": np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64),

            # right face: face +Z looks along camera +X
            "pos_x": np.array([
                [ 0.0, 0.0, 1.0],
                [ 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float64),

            # left face: face +Z looks along camera -X
            "neg_x": np.array([
                [0.0, 0.0, -1.0],
                [0.0, 1.0,  0.0],
                [1.0, 0.0,  0.0],
            ], dtype=np.float64),

            # lower face in image coordinates: face +Z looks along camera +Y
            "pos_y": np.array([
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0],
                [0.0, -1.0, 0.0],
            ], dtype=np.float64),

            # upper face in image coordinates: face +Z looks along camera -Y
            "neg_y": np.array([
                [1.0, 0.0,  0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0,  0.0],
            ], dtype=np.float64),

            # Available for debugging only. The 224 KB4 observation normally
            # should not need it with this calibration.
            "neg_z": np.array([
                [-1.0, 0.0,  0.0],
                [ 0.0, 1.0,  0.0],
                [ 0.0, 0.0, -1.0],
            ], dtype=np.float64),
        }

    def print_panda_link_info(self):
        """Print PyBullet joint/link indices to verify eye_parent_link."""
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            info = self.bullet_client.getJointInfo(self.panda, j)
            joint_name = info[1].decode("utf-8") if isinstance(info[1], bytes) else str(info[1])
            link_name = info[12].decode("utf-8") if isinstance(info[12], bytes) else str(info[12])
            print(f"joint/link index {j}: joint={joint_name}, link={link_name}")

    def pose_to_T(self, pos, quat_xyzw):
        """Convert a PyBullet pose into a 4x4 transform."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(
            self.bullet_client.getMatrixFromQuaternion(quat_xyzw),
            dtype=np.float64,
        ).reshape(3, 3)
        T[:3, 3] = np.asarray(pos, dtype=np.float64)
        return T

    def get_eye_in_hand_T_world_cam(self):
        """Return the current eye-in-hand camera pose in world frame.

        Assumed calibration direction:
            T_world_cam = T_world_eye_parent @ T_eye_parent_cam

        If your calibration file actually stores cam->parent instead of
        parent->cam, replace self.T_eye_parent_cam below with
        np.linalg.inv(self.T_eye_parent_cam).
        """
        link_state = self.bullet_client.getLinkState(
            self.panda,
            self.eye_parent_link,
            computeForwardKinematics=True,
        )

        parent_pos = np.array(link_state[4], dtype=np.float64)
        parent_quat = np.array(link_state[5], dtype=np.float64)
        T_world_parent = self.pose_to_T(parent_pos, parent_quat)

        # debug_euler_xyz_deg = np.array([90, 0, 0], dtype=np.float64)

        # debug_rot = np.eye(4, dtype=np.float64)
        # debug_rot[:3, :3] = R.from_euler(
        #     "xyz",
        #     debug_euler_xyz_deg,
        #     degrees=True,
        # ).as_matrix()

        # T_world_parent = T_world_parent @ debug_rot
        # # Right-multiply: rotate the camera frame around its own local x/y/z axes.
        # self.debug_draw_axes(T_world_parent, length= 0.1, life_time= 200000)
        return T_world_parent @ self.T_eye_parent_cam

    def kb4_fisheye_rays_for_obs(self, out_width, out_height):
        """Return 224x224 observation rays using the original 640x480 KB4 model.

        The policy image is treated as a resize of the real 640x480 fisheye
        frame into out_width x out_height. We therefore map each output pixel
        center back to the corresponding raw 640x480 pixel coordinate before
        applying the KB4 inverse projection.
        """
        fx = float(self.eye_K[0, 0])
        fy = float(self.eye_K[1, 1])
        cx = float(self.eye_K[0, 2])
        cy = float(self.eye_K[1, 2])
        k1, k2, k3, k4 = [float(x) for x in self.eye_D]

        u_obs, v_obs = np.meshgrid(
            np.arange(out_width, dtype=np.float64),
            np.arange(out_height, dtype=np.float64),
        )

        # Equivalent pixel-center mapping for cv2.resize from 640x480 to 224x224.
        u_raw = (u_obs + 0.5) * (self.eye_raw_width / float(out_width)) - 0.5
        v_raw = (v_obs + 0.5) * (self.eye_raw_height / float(out_height)) - 0.5

        mx = (u_raw - cx) / fx
        my = (v_raw - cy) / fy

        theta_d = np.sqrt(mx * mx + my * my)
        phi = np.arctan2(my, mx)

        # Newton solve for theta in:
        # theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
        theta = theta_d.copy()
        for _ in range(8):
            t2 = theta * theta
            t4 = t2 * t2
            t6 = t4 * t2
            t8 = t4 * t4
            f = theta * (1.0 + k1*t2 + k2*t4 + k3*t6 + k4*t8) - theta_d
            df = 1.0 + 3.0*k1*t2 + 5.0*k2*t4 + 7.0*k3*t6 + 9.0*k4*t8
            theta = theta - f / np.maximum(np.abs(df), 1e-12)

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        rays = np.empty((out_height, out_width, 3), dtype=np.float64)
        rays[..., 0] = sin_t * np.cos(phi)
        rays[..., 1] = sin_t * np.sin(phi)
        rays[..., 2] = cos_t

        center_mask = theta_d < 1e-12
        rays[center_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1e-12)
        return rays

    def build_eye_fisheye_remap(self, out_width=224, out_height=224, face_size=256):
        """Precompute all KB4 fisheye -> cubemap face remap arrays.

        Runtime rendering then only needs to render the five faces and call
        cv2.remap with these cached maps.
        """
        rays = self.kb4_fisheye_rays_for_obs(out_width, out_height)
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

        face_idx = -np.ones((out_height, out_width), dtype=np.int32)
        face_to_idx = {name: i for i, name in enumerate(self.eye_face_names)}

        # Choose the cubemap face whose forward axis is closest to the ray.
        if "pos_z" in face_to_idx:
            face_idx[(az >= ax) & (az >= ay) & (z >= 0.0)] = face_to_idx["pos_z"]
        if "pos_x" in face_to_idx:
            face_idx[(ax > az) & (ax >= ay) & (x >= 0.0)] = face_to_idx["pos_x"]
        if "neg_x" in face_to_idx:
            face_idx[(ax > az) & (ax >= ay) & (x < 0.0)] = face_to_idx["neg_x"]
        if "pos_y" in face_to_idx:
            face_idx[(ay > ax) & (ay > az) & (y >= 0.0)] = face_to_idx["pos_y"]
        if "neg_y" in face_to_idx:
            face_idx[(ay > ax) & (ay > az) & (y < 0.0)] = face_to_idx["neg_y"]
        if "neg_z" in face_to_idx:
            face_idx[(az >= ax) & (az >= ay) & (z < 0.0)] = face_to_idx["neg_z"]

        remap = {}
        for name in self.eye_face_names:
            idx = face_to_idx[name]
            mask = face_idx == idx
            R_cam_face = self.eye_face_basis[name]

            # row-vector form: ray_face = R_cam_face.T @ ray_cam
            ray_face = rays @ R_cam_face
            rz = ray_face[..., 2]
            valid = mask & (rz > 1e-8)

            xn = ray_face[..., 0] / np.maximum(rz, 1e-8)
            yn = ray_face[..., 1] / np.maximum(rz, 1e-8)

            # For a 90-degree face, normalized coordinates are in [-1, 1].
            map_x = ((xn + 1.0) * 0.5 * (face_size - 1)).astype(np.float32)
            map_y = ((yn + 1.0) * 0.5 * (face_size - 1)).astype(np.float32)

            # Invalid pixels are remapped to black but will not be copied anyway.
            map_x[~valid] = 0.0
            map_y[~valid] = 0.0

            remap[name] = {
                "mask": valid,
                "map_x": map_x,
                "map_y": map_y,
            }

        self.eye_invalid_pixel_count = int(np.sum(face_idx < 0))
        return remap

    def get_eye_face_view_matrix(self, T_world_cam, face_name):
        """Build a PyBullet view matrix for one cubemap face."""
        T_cam_face = np.eye(4, dtype=np.float64)
        T_cam_face[:3, :3] = self.eye_face_basis[face_name]
        T_world_face = T_world_cam @ T_cam_face
        return cvPose2BulletView(T_world_face)

    def render_eye_cubemap_faces(self):
        """Render the low-res pinhole faces used to synthesize the KB4 image."""
        T_world_cam = self.get_eye_in_hand_T_world_cam()
        faces = {}
        for face_name in self.eye_face_names:
            view = self.get_eye_face_view_matrix(T_world_cam, face_name)
            _, _, rgba, _, _ = self.render_camera_raw(
                self.eye_face_size,
                self.eye_face_size,
                view,
                self.eye_face_proj_90,
            )
            faces[face_name] = rgba[..., :3]
        return faces

    def debug_draw_axes(self, T, length=0.08, life_time=0.1):
        """Draw eye camera axes in the PyBullet GUI for calibration debugging."""
        o = T[:3, 3]
        Rwc = T[:3, :3]
        self.bullet_client.addUserDebugLine(o, o + length * Rwc[:, 0], [1, 0, 0], 2, life_time)
        self.bullet_client.addUserDebugLine(o, o + length * Rwc[:, 1], [0, 1, 0], 2, life_time)
        self.bullet_client.addUserDebugLine(o, o + length * Rwc[:, 2], [0, 0, 1], 2, life_time)

    def render_camera_raw(self, width, height, view_matrix, proj_matrix):
        """Render a camera without image noise.

        This is used by the fisheye cubemap renderer. Noise is applied once
        after the five faces are remapped into a single fisheye image, avoiding
        visible seams between cubemap faces.
        """
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

        bg_mask = seg < 0
        rgba[bg_mask, :3] = 0
        rgba[bg_mask, 3] = 255

        return w, h, rgba, depth, seg

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
        rgb = self.agentviewImgDR.apply_image_noise(rgb)
        rgba[..., :3] = rgb


        return w, h, rgba, depth, seg

    def get_cropped_agentview_image(self, out_size=224):
        _, _, rgba, _, _ = self.get_agentview_image()
        rgb = rgba[..., :3]
        cropped_rgb = resize_rgb(rgb, out_size)
        return cropped_rgb

    def get_agentview_image(self):
        projectionMatrix = cvK2BulletP(self.agentview_intrinsic,
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
        intrinsic_960x540 = self.agentview_intrinsic.copy()

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

  
    def get_eye_in_hand_image(self, width=None, height=None, face_size=None):
        """Return the eye-in-hand KB4/equidistant fisheye observation.

        Default output is 224x224 for Diffusion Policy training. The image is
        generated by rendering five low-resolution 90-degree pinhole faces and
        remapping them with the precomputed KB4 model derived from the original
        640x480 calibration.
        """
        if width is None:
            width = self.eye_obs_width
        if height is None:
            height = self.eye_obs_height

        # Fast path: default precomputed remap.
        if (
            int(width) == self.eye_obs_width
            and int(height) == self.eye_obs_height
            and (face_size is None or int(face_size) == self.eye_face_size)
        ):
            remap = self.eye_fisheye_remap
            old_face_size = self.eye_face_size
        else:
            # Debug path for non-default sizes. This recomputes maps and should
            # not be used inside high-throughput training loops.
            old_face_size = self.eye_face_size
            if face_size is not None:
                self.eye_face_size = int(face_size)
            remap = self.build_eye_fisheye_remap(
                out_width=int(width),
                out_height=int(height),
                face_size=self.eye_face_size,
            )

        faces = self.render_eye_cubemap_faces()
        out = np.zeros((int(height), int(width), 3), dtype=np.uint8)

        for face_name in self.eye_face_names:
            face = faces[face_name]
            m = remap[face_name]
            sampled = cv2.remap(
                face,
                m["map_x"],
                m["map_y"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = m["mask"]
            out[mask] = sampled[mask]

        if self.eye_invalid_pixel_count > 0:
            # This should be 0 with the current 5-face setup. If not, switch
            # self.eye_face_names to include "neg_z" and rebuild the remap.
            pass

        # Apply eye-camera image DR once after fisheye composition.
        # Do not apply it per cubemap face, otherwise color/noise seams can show up.
        out = self.eyeImgDR.apply_image_noise(out)

        # Restore face size if debug path changed it.
        self.eye_face_size = old_face_size
        return out.astype(np.uint8)


    def _quat_axis(self, quat_xyzw, local_axis):
        quat_xyzw = np.asarray(quat_xyzw, dtype=float)
        local_axis = np.asarray(local_axis, dtype=float)
        qn = np.linalg.norm(quat_xyzw)
        an = np.linalg.norm(local_axis)
        if qn < 1e-12 or an < 1e-12:
            return None
        axis = R.from_quat(quat_xyzw / qn).apply(local_axis / an)
        norm = np.linalg.norm(axis)
        return None if norm < 1e-8 else axis / norm

    def get_assembly_alignment_metrics(self):
        """Measure ring-center/rod-center alignment in the child coordinate frame."""
        if not hasattr(self, "assembly_parent_id") or not hasattr(self, "assembly_child_id"):
            return None

        parent_pos, parent_orn = self.get_parent_obj_pose()
        target_pos, target_orn = self.get_assembly_target_parent_pose()
        _, child_orn = self.get_child_obj_pose()

        axis_world = self._quat_axis(child_orn, self.assembly_axis_local)
        parent_axis = self._quat_axis(parent_orn, self.assembly_axis_local)
        target_axis = self._quat_axis(target_orn, self.assembly_axis_local)
        if axis_world is None or parent_axis is None or target_axis is None:
            return None

        delta = parent_pos - target_pos
        axial_error = float(np.dot(delta, axis_world))
        radial_vec = delta - axial_error * axis_world
        radial_error = float(np.linalg.norm(radial_vec))

        # A ring axis is sign-symmetric, so +axis and -axis are equivalent.
        axis_error_rad = float(
            np.arccos(np.clip(abs(np.dot(parent_axis, target_axis)), -1.0, 1.0))
        )
        orientation_error_rad = float(
            np.linalg.norm(
                (R.from_quat(target_orn).inv() * R.from_quat(parent_orn)).as_rotvec()
            )
        )
        return {
            "radial_error": radial_error,
            "axial_error": axial_error,
            "axis_error_rad": axis_error_rad,
            "orientation_error_rad": orientation_error_rad,
            "parent_pos": parent_pos,
            "target_pos": target_pos,
        }

    def has_parent_child_contact(self, max_contact_distance=0.003):
        contacts = self.bullet_client.getContactPoints(
            bodyA=self.assembly_parent_id,
            bodyB=self.assembly_child_id,
        )
        return any(c[8] <= float(max_contact_distance) for c in contacts)

    def is_parent_grasped(self):
        if self.parent_grasp_constraint_id is not None:
            return True
        contacts = self.bullet_client.getContactPoints(
            bodyA=self.panda,
            bodyB=self.assembly_parent_id,
        )
        return len(contacts) > 0 and self.get_gripper_mean_width() < 0.03

    def get_gripper_mean_width(self):
        finger_widths = np.array([
            self.bullet_client.getJointState(self.panda, 9)[0],
            self.bullet_client.getJointState(self.panda, 10)[0],
        ], dtype=float)
        return float(np.mean(finger_widths))

    def is_success(self,
                   radial_tol=0.018,
                   axial_tol=0.025,
                   axis_tol_deg=15.0,
                   require_grasp=True,
                   require_parent_child_contact=False,
                   require_done=False,
                   return_info=False,
                   debug=False):
        """Check whether the parent ring is centered and inserted over the child rod."""
        info = {"success": False, "reason": None}

        def finish(value):
            info["success"] = bool(value)
            if debug:
                print("\n========== [AssemblySim is_success] ==========")
                for key, val in info.items():
                    print(f"{key}: {val}")
                print("==============================================\n")
            return (bool(value), info) if return_info else bool(value)

        if not hasattr(self, "assembly_parent_id"):
            info["reason"] = "missing_parent"
            return finish(False)
        if not hasattr(self, "assembly_child_id"):
            info["reason"] = "missing_child"
            return finish(False)
        if require_done and not self.done:
            info["reason"] = "state_machine_not_done"
            return finish(False)

        metrics = self.get_assembly_alignment_metrics()
        if metrics is None:
            info["reason"] = "invalid_alignment_metrics"
            return finish(False)

        info.update({
            "radial_error": metrics["radial_error"],
            "axial_error": metrics["axial_error"],
            "axis_error_deg": math.degrees(metrics["axis_error_rad"]),
            "radial_tol": float(radial_tol),
            "axial_tol": float(axial_tol),
            "axis_tol_deg": float(axis_tol_deg),
        })

        if metrics["radial_error"] > float(radial_tol):
            info["reason"] = "parent_not_centered_on_child"
            return finish(False)
        if abs(metrics["axial_error"]) > float(axial_tol):
            info["reason"] = "parent_not_at_insertion_depth"
            return finish(False)
        if metrics["axis_error_rad"] > math.radians(float(axis_tol_deg)):
            info["reason"] = "parent_child_axis_misaligned"
            return finish(False)

        if require_grasp:
            grasped = self.is_parent_grasped()
            info["parent_grasped"] = bool(grasped)
            if not grasped:
                info["reason"] = "parent_not_grasped"
                return finish(False)

        if require_parent_child_contact:
            has_contact = self.has_parent_child_contact()
            info["parent_child_contact"] = bool(has_contact)
            if not has_contact:
                info["reason"] = "no_parent_child_contact"
                return finish(False)

        info["reason"] = "success"
        return finish(True)

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