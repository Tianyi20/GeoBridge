import math
import cv2
import numpy as np
import pybullet_data
from scipy.spatial.transform import Rotation as R

from VisualDR import (
    DistractorDR,
    ImgNoiseDR,
    IntrinsicDR,
    LightingDR,
    ObjectColorDR,
    PoseDR,
    TextureDR,
)
from pybullet_utility import (
    coacd_convex_decomposition,
    cvK2BulletP,
    cvPose2BulletView,
    get_com,
    get_true_PositionAndOrientation,
    load_models,
    quat_slerp,
)
from utility import (
    load_initial_grasp_pose,
    quat_from_rotation_matrix,
    resize_rgb,
)


pandaEndEffectorIndex = 11
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
ul = [7] * pandaNumDofs
jr = [7] * pandaNumDofs

jointPositions = [
    -1.5181891389821258,
    1.5913041972968864,
    1.5482446948770892,
    -1.6774729466856573,
    -0.0008096577587373927,
    1.755319486025307,
    0.8361516126271482,
    0.04,
    0.04,
]


class GearHorizonSim(object):
    """Extract the currently exposed gear from a one-to-three gear stack."""

    def __init__(
        self,
        bullet_client,
        offset,
        control_dt=1.0 / 120.0,
        seed=42,
        randomize_initial_ee_pose=True,
        initial_ee_x_jit=0.05,
        initial_ee_y_jit=0.04,
        initial_ee_z_jit=0.05,
        initial_ee_eul_jit=0.12,
    ):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(
            solverResidualThreshold=0
        )

        self.lightingDR = LightingDR(self.bullet_client, seed=seed)
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
        self.ImgNoiseDR = self.agentviewImgDR

        self.gearPoseDR = PoseDR(self.bullet_client, seed=seed)
        self.supportorPoseDR = PoseDR(self.bullet_client, seed=seed + 10)
        self.objposeDR = self.gearPoseDR
        self.camposeDR = PoseDR(self.bullet_client, seed=seed)
        self.outsceneDR = PoseDR(self.bullet_client, seed=seed)
        self.collplaneDR = PoseDR(self.bullet_client, seed=seed)
        self.distractorDR = DistractorDR(self.bullet_client, seed=seed)
        self.objectColorDR = ObjectColorDR(self.bullet_client, seed=seed)
        self.robotTextureDR = TextureDR(
            self.bullet_client,
            seed=seed + 4000,
        )
        self.fisheyeCamDR = PoseDR(self.bullet_client, seed=seed)
        self.initialEePoseDR = PoseDR(
            self.bullet_client,
            seed=seed + 2000,
        )
        self.camIntrinsicDR = IntrinsicDR(seed=seed + 3000)
        self.objectInHandPoseDR = PoseDR(
            self.bullet_client,
            seed=seed + 5000,
        )
        self.sceneRng = np.random.default_rng(seed + 6000)

        self.offset = np.asarray(offset, dtype=float)
        self.control_dt = float(control_dt)
        self.t = 0.0
        self.bullet_client.configureDebugVisualizer(
            rgbBackground=[1, 1, 1]
        )

        self.agentview_width = 960
        self.agentview_height = 540
        self.agentview_near = 0.02
        self.agentview_far = 2.0
        self.agentview_base_extrinsic_cam = np.array([
            [-0.808,   0.3283, -0.4892,  0.8758],
            [ 0.5837,  0.3327, -0.7407,  0.6006],
            [-0.0804, -0.8840, -0.4604,  0.4729],
            [ 0.0,     0.0,     0.0,     1.0],
        ], dtype=np.float32)
        self.agentview_base_intrinsic = np.array([
            [691.7508,   0.0,     486.7637],
            [  0.0,    692.2195, 273.4784],
            [  0.0,      0.0,       1.0],
        ], dtype=np.float32)
        self.agentview_intrinsic = (
            self.agentview_base_intrinsic.copy()
        )

        self.eye_raw_width = 640
        self.eye_raw_height = 480
        self.eye_obs_width = 224
        self.eye_obs_height = 224
        self.eye_face_size = 256
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
        self.T_eye_base_parent_cam = np.array([
            [0, -1, 0,  0.05054945],
            [1,  0, 0, -0.00619893],
            [0,  0, 1,  0.01294445],
            [0,  0, 0,  1.0],
        ], dtype=np.float64)
        self.eye_parent_link = 8
        self.eye_face_names = (
            "pos_z",
            "pos_x",
            "neg_x",
            "pos_y",
            "neg_y",
        )
        self.eye_face_basis = self._build_eye_face_basis()
        self.eye_face_proj_90 = (
            self.bullet_client.computeProjectionMatrixFOV(
                fov=90.0,
                aspect=1.0,
                nearVal=self.eye_near,
                farVal=self.eye_far,
            )
        )
        self.eye_fisheye_remap = self.build_eye_fisheye_remap(
            out_width=self.eye_obs_width,
            out_height=self.eye_obs_height,
            face_size=self.eye_face_size,
        )

        self.GRIPPER_CLOSED = 0.0
        self.GRIPPER_OPEN = 1.0
        self.gripper_closed_width = 0.0
        self.gripper_open_width = 0.04
        self.gripper_state_threshold_width = 0.8 * (
            self.gripper_open_width + self.gripper_closed_width
        )
        self.target_gripper = self.GRIPPER_OPEN
        self.finger_target = self.gripper_open_width

        self.safe_approach = 0.08
        self.lift_height = 0.14
        self.place_approach_height = 0.10
        self.retreat_height = 0.10
        self.arm_force = 200.0
        self.gripper_force = 200.0

        flags = (
            self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        base_orn = self.bullet_client.getQuaternionFromEuler(
            [0, 0, 0]
        )
        self.panda = self.bullet_client.loadURDF(
            "franka_panda/panda_wristcam.urdf",
            np.array([0.0, 0.0, 0.0]) + self.offset,
            base_orn,
            useFixedBase=True,
            flags=flags,
        )
        finger_constraint = self.bullet_client.createConstraint(
            self.panda,
            9,
            self.panda,
            10,
            jointType=self.bullet_client.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.bullet_client.changeConstraint(
            finger_constraint,
            gearRatio=-1,
            erp=0.1,
            maxForce=100,
        )

        index = 0
        for joint_idx in range(
            self.bullet_client.getNumJoints(self.panda)
        ):
            self.bullet_client.changeDynamics(
                self.panda,
                joint_idx,
                linearDamping=0,
                angularDamping=0,
            )
            joint_type = self.bullet_client.getJointInfo(
                self.panda,
                joint_idx,
            )[2]
            if joint_type in (
                self.bullet_client.JOINT_PRISMATIC,
                self.bullet_client.JOINT_REVOLUTE,
            ):
                self.bullet_client.resetJointState(
                    self.panda,
                    joint_idx,
                    jointPositions[index],
                )
                index += 1

        base_ee_pos, base_ee_orn = self.get_ee_pose()
        if randomize_initial_ee_pose:
            init_ee_pos, init_ee_orn = (
                self.initialEePoseDR.sample_SE3_randomization(
                    pos=base_ee_pos,
                    orn=base_ee_orn,
                    x_jitter_range=initial_ee_x_jit,
                    y_jitter_range=initial_ee_y_jit,
                    z_jitter_range=initial_ee_z_jit,
                    x_euler_jitter_range=initial_ee_eul_jit,
                    y_euler_jitter_range=initial_ee_eul_jit,
                    z_euler_jitter_range=initial_ee_eul_jit,
                )
            )
            self.solve_ik_and_apply(
                init_ee_pos,
                init_ee_orn,
                reset=True,
            )

        self.home_joint = np.asarray(
            self.get_current_arm_joints(),
            dtype=float,
        )
        self.home_ee_pos, self.home_ee_orn = self.get_ee_pose()
        self.home_ee_pos = np.asarray(
            self.home_ee_pos,
            dtype=float,
        )
        self.home_ee_orn = np.asarray(
            self.home_ee_orn,
            dtype=float,
        )

        self.states = [
            "home",
            "move_pregrasp",
            "open_gripper",
            "move_grasp",
            "close_gripper",
            "lift_gear",
            "move_above_place",
            "lower_to_place",
            "open_gripper",
            "retreat",
        ]
        self.state_durations = [
            0.01,
            3.0,
            0.25,
            1.5,
            0.5,
            1.0,
            3.0,
            1.5,
            0.25,
            0.2,
        ]
        self.state_idx = 0
        self.state = self.states[0]
        self.state_t = 0.0
        self.done = False

        self.target_pos = None
        self.target_orn = None
        self.motion_start_pos = None
        self.motion_start_orn = None
        self.motion_target_pos = None
        self.motion_target_orn = None
        self.gear_motion_start_pos = None
        self.gear_motion_start_orn = None
        self.gear_motion_target_pos = None
        self.gear_motion_target_orn = None

        self.last_grasp_pose = None
        self.last_grasp_orn = None
        self.initial_gear_pos = None
        self.initial_gear_orn = None
        self.initial_supportor_pos = None
        self.initial_supportor_orn = None

        self.gear_grasp_constraint_id = None
        self.grasp_gear_to_ee = None
        self.nominal_grasp_finger_width = None
        self.hold_nominal_gripper_width_after_attach = True
        self.nominal_ee_to_gear = None
        self.randomized_ee_to_gear = None
        self.object_in_hand_delta_pos = np.zeros(
            3,
            dtype=float,
        )
        self.object_in_hand_delta_euler = np.zeros(
            3,
            dtype=float,
        )

        self.fix_gear_to_gripper = True
        self.randomize_object_in_hand_pose = True
        self.object_in_hand_pos_jit = np.array(
            [0.002, 0.002, 0.003],
            dtype=float,
        )
        self.object_in_hand_eul_jit = np.array(
            [0.0174533, 0.0174533, 0.0349066],
            dtype=float,
        )
        self.object_in_hand_debug = False
        self.disable_gear_gripper_collision = True
        self.gear_gripper_collision_links = (
            8,
            9,
            10,
            pandaEndEffectorIndex,
        )
        self.gear_gripper_collision_disabled = False

        self.gear_to_support_pos = np.array(
            [0.0, 0.0, 0.31],
            dtype=float,
        )
        self.gear_to_support_orn = np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=float,
        )
        self.place_gear_pos = None
        self.place_gear_orn = None
        self.transport_gear_z = None

        self.prepare_state(self.state)

    def get_gear_pose(self):
        return tuple(
            np.asarray(v, dtype=float)
            for v in get_true_PositionAndOrientation(
                self.bullet_client, self.gear_id
            )
        )

    def get_supportor_pose(self):
        return tuple(
            np.asarray(v, dtype=float)
            for v in get_true_PositionAndOrientation(
                self.bullet_client, self.supportor_id
            )
        )

    def get_gear_to_ee_transform(self, ee_pos=None, ee_orn=None):
        """Return the current gear-frame -> EE transform.

        With no explicit EE pose, this reads both bodies from PyBullet every
        time, so grasp slip is reflected immediately. An explicit EE pose is
        only used when building annotation-based debug/distractor waypoints.
        """
        gear_pos, gear_orn = self.get_gear_pose()
        if ee_pos is None or ee_orn is None:
            ee_pos, ee_orn = self.get_ee_pose()

        inv_gear_pos, inv_gear_orn = self.bullet_client.invertTransform(
            gear_pos.tolist(), gear_orn.tolist()
        )
        rel_pos, rel_orn = self.bullet_client.multiplyTransforms(
            inv_gear_pos,
            inv_gear_orn,
            np.asarray(ee_pos, dtype=float).tolist(),
            np.asarray(ee_orn, dtype=float).tolist(),
        )
        return np.asarray(rel_pos, dtype=float), np.asarray(rel_orn, dtype=float)

    def set_gear_gripper_collision_enabled(self, enabled):
        """Enable/disable only gear-vs-local-gripper collision pairs.

        ``JOINT_FIXED`` constrains relative motion but does not suppress contact
        generation.  This helper therefore filters collision only between the
        gear's base link and the configured Panda hand/finger/EE links.
        Gear collisions with the supportor, plane, environment and distractors are
        intentionally untouched.
        """
        panda_id = getattr(self, "panda", None)
        gear_id = getattr(self, "gear_id", None)
        if panda_id is None or gear_id is None:
            self.gear_gripper_collision_disabled = False
            return

        enable_flag = 1 if bool(enabled) else 0
        num_robot_joints = self.bullet_client.getNumJoints(panda_id)
        configured_links = getattr(
            self,
            "gear_gripper_collision_links",
            (8, 9, 10, pandaEndEffectorIndex),
        )

        for robot_link in configured_links:
            robot_link = int(robot_link)
            if robot_link < -1 or robot_link >= num_robot_joints:
                if getattr(self, "object_in_hand_debug", False):
                    print(
                        "Skipping invalid gear/gripper collision link:",
                        robot_link,
                    )
                continue
            self.bullet_client.setCollisionFilterPair(
                bodyUniqueIdA=panda_id,
                bodyUniqueIdB=gear_id,
                linkIndexA=robot_link,
                linkIndexB=-1,
                enableCollision=enable_flag,
            )

        self.gear_gripper_collision_disabled = not bool(enabled)

    def remove_gear_grasp_constraint(self):
        """Remove rigid attachment and restore gear/gripper collisions."""
        constraint_id = getattr(self, "gear_grasp_constraint_id", None)
        if constraint_id is not None:
            try:
                self.bullet_client.removeConstraint(constraint_id)
            except Exception:
                pass
        self.gear_grasp_constraint_id = None
        self.grasp_gear_to_ee = None
        self.nominal_grasp_finger_width = None

        # Restore only the pairs that this class may have disabled.
        try:
            self.set_gear_gripper_collision_enabled(True)
        except Exception:
            # This can happen during scene teardown after a body was already
            # removed.  The next loaded body starts with collisions enabled.
            self.gear_gripper_collision_disabled = False

    def get_ee_to_gear_transform(self, ee_pos=None, ee_orn=None):
        """Return the gear pose expressed in the EE link frame."""
        gear_pos, gear_orn = self.get_gear_pose()
        if ee_pos is None or ee_orn is None:
            ee_pos, ee_orn = self.get_ee_pose()

        inv_ee_pos, inv_ee_orn = self.bullet_client.invertTransform(
            np.asarray(ee_pos, dtype=float).tolist(),
            np.asarray(ee_orn, dtype=float).tolist(),
        )
        rel_pos, rel_orn = self.bullet_client.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            gear_pos.tolist(),
            gear_orn.tolist(),
        )
        return np.asarray(rel_pos, dtype=float), np.asarray(rel_orn, dtype=float)

    def randomize_and_fix_gear_to_gripper(self):
        """Sample an object-in-hand SE(3) pose and rigidly attach the gear.

        The nominal grasp trajectory is executed first. At the end of
        ``close_gripper`` we measure the actual gear pose in the EE
        frame, add a small translation/orientation perturbation in that frame,
        teleport the gear to the sampled pose, and create a fixed constraint.

        No virtual force or physical slip model is used. The purpose is to
        generate image/proprioception-to-action mappings under varied
        object-in-hand poses.
        """
        if not self.fix_gear_to_gripper:
            self.grasp_gear_to_ee = None
            return

        self.remove_gear_grasp_constraint()

        # close_gripper has just completed with normal gear/finger collision.
        # Capture the physically achieved nominal grasp width before teleporting
        # the gear, filtering collision, or creating the rigid attachment.
        # The current simulator convention uses each finger joint position as
        # the gripper width command, so use the symmetric mean as the hold target.
        nominal_finger_qpos = np.array([
            self.bullet_client.getJointState(self.panda, 9)[0],
            self.bullet_client.getJointState(self.panda, 10)[0],
        ], dtype=float)
        self.nominal_grasp_finger_width = float(np.mean(nominal_finger_qpos))

        ee_pos, ee_orn = self.get_ee_pose()
        gear_true_pos, gear_true_orn = self.get_gear_pose()
        gear_base_pos, gear_base_orn = self.bullet_client.getBasePositionAndOrientation(
            self.gear_id
        )
        gear_base_pos = np.asarray(gear_base_pos, dtype=float)
        gear_base_orn = np.asarray(gear_base_orn, dtype=float)

        # Nominal object pose in EE coordinates, based on the object/mesh frame
        # used by the gear target calculations.
        nominal_rel_pos, nominal_rel_orn = self.get_ee_to_gear_transform(
            ee_pos=ee_pos, ee_orn=ee_orn
        )
        self.nominal_ee_to_gear = (
            nominal_rel_pos.copy(),
            nominal_rel_orn.copy(),
        )

        if self.randomize_object_in_hand_pose:
            randomized_rel_pos, randomized_rel_orn = (
                self.objectInHandPoseDR.sample_SE3_randomization(
                    pos=nominal_rel_pos,
                    orn=nominal_rel_orn,
                    x_jitter_range=self.object_in_hand_pos_jit[0],
                    y_jitter_range=self.object_in_hand_pos_jit[1],
                    z_jitter_range=self.object_in_hand_pos_jit[2],
                    x_euler_jitter_range=self.object_in_hand_eul_jit[0],
                    y_euler_jitter_range=self.object_in_hand_eul_jit[1],
                    z_euler_jitter_range=self.object_in_hand_eul_jit[2],
                )
            )
            randomized_rel_pos = np.asarray(randomized_rel_pos, dtype=float)
            randomized_rel_orn = np.asarray(randomized_rel_orn, dtype=float)
        else:
            randomized_rel_pos = nominal_rel_pos.copy()
            randomized_rel_orn = nominal_rel_orn.copy()

        delta_pos = randomized_rel_pos - nominal_rel_pos
        delta_euler = (
            R.from_quat(nominal_rel_orn).inv()
            * R.from_quat(randomized_rel_orn)
        ).as_euler("xyz")

        self.object_in_hand_delta_pos = delta_pos.copy()
        self.object_in_hand_delta_euler = delta_euler.copy()
        self.randomized_ee_to_gear = (
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
        # constraint operate on the base frame, while gear pose calculations use the
        # true/mesh frame returned by get_true_PositionAndOrientation().
        inv_true_pos, inv_true_orn = self.bullet_client.invertTransform(
            gear_true_pos.tolist(), gear_true_orn.tolist()
        )
        true_to_base_pos, true_to_base_orn = self.bullet_client.multiplyTransforms(
            inv_true_pos,
            inv_true_orn,
            gear_base_pos.tolist(),
            gear_base_orn.tolist(),
        )
        randomized_base_pos, randomized_base_orn = self.bullet_client.multiplyTransforms(
            randomized_true_pos,
            randomized_true_orn,
            true_to_base_pos,
            true_to_base_orn,
        )

        self.bullet_client.resetBasePositionAndOrientation(
            self.gear_id,
            randomized_base_pos,
            randomized_base_orn,
        )
        self.bullet_client.resetBaseVelocity(
            self.gear_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

        # Fixed-constraint robot frame is the EE link; child frame is the
        # PyBullet gear base.
        inv_ee_pos, inv_ee_orn = self.bullet_client.invertTransform(
            ee_pos.tolist(), ee_orn.tolist()
        )
        ee_to_base_pos, ee_to_base_orn = self.bullet_client.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            randomized_base_pos,
            randomized_base_orn,
        )
        # JOINT_FIXED does not automatically ignore collisions.  Disable only
        # gear-vs-gripper-neighborhood contacts before creating the rigid
        # attachment; all other gear collisions remain enabled.
        if self.disable_gear_gripper_collision:
            self.set_gear_gripper_collision_enabled(False)
        else:
            self.set_gear_gripper_collision_enabled(True)

        try:
            self.gear_grasp_constraint_id = self.bullet_client.createConstraint(
                parentBodyUniqueId=self.panda,
                parentLinkIndex=pandaEndEffectorIndex,
                childBodyUniqueId=self.gear_id,
                childLinkIndex=-1,
                jointType=self.bullet_client.JOINT_FIXED,
                jointAxis=[0.0, 0.0, 0.0],
                parentFramePosition=ee_to_base_pos,
                childFramePosition=[0.0, 0.0, 0.0],
                parentFrameOrientation=ee_to_base_orn,
                childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            )
        except Exception:
            # Do not leave collision filtering or a stale width hold active if
            # attachment creation fails.
            self.set_gear_gripper_collision_enabled(True)
            self.nominal_grasp_finger_width = None
            raise

        # Immediately replace the old fully-closed motor target with the width
        # achieved during nominal physical grasping. Subsequent calls to
        # set_gripper_state(CLOSED) will keep re-applying this same target.
        if self.hold_nominal_gripper_width_after_attach:
            self.set_gripper_width(self.nominal_grasp_finger_width)

        # Cache gear -> EE for all subsequent expert targets.
        cached_pos, cached_orn = self.bullet_client.invertTransform(
            randomized_rel_pos.tolist(), randomized_rel_orn.tolist()
        )
        self.grasp_gear_to_ee = (
            np.asarray(cached_pos, dtype=float),
            np.asarray(cached_orn, dtype=float),
        )

        if self.object_in_hand_debug:
            print(
                "object-in-hand DR:",
                "delta_pos_ee=", np.round(delta_pos, 6),
                "delta_euler_deg=", np.round(np.degrees(delta_euler), 3),
                "constraint_id=", self.gear_grasp_constraint_id,
                "gear_gripper_collision_disabled=",
                self.gear_gripper_collision_disabled,
                "filtered_robot_links=",
                self.gear_gripper_collision_links,
                "nominal_grasp_finger_width=",
                round(float(self.nominal_grasp_finger_width), 6),
            )

    def get_ee_pose_for_gear_pose(
        self, gear_pos, gear_orn, gear_to_ee=None
    ):
        """Convert a desired gear pose into an EE pose.

        Runtime transport calls leave ``gear_to_ee`` unset and re-measure the
        transform every step. After rigid attachment this transform should be
        constant apart from tiny solver noise, matching the continuously
        measured gear-to-EE mapping used by the scripted expert.
        """
        if gear_to_ee is None:
            gear_to_ee = self.get_gear_to_ee_transform()
        rel_pos, rel_orn = gear_to_ee
        ee_pos, ee_orn = self.bullet_client.multiplyTransforms(
            np.asarray(gear_pos, dtype=float).tolist(),
            np.asarray(gear_orn, dtype=float).tolist(),
            np.asarray(rel_pos, dtype=float).tolist(),
            np.asarray(rel_orn, dtype=float).tolist(),
        )
        return np.asarray(ee_pos, dtype=float), np.asarray(ee_orn, dtype=float)

    def make_scene(
        self,
        env_mesh_path=None,
        gear_assets=None,
        supportor_path="./data/objects/gear_extraction/supportor/supportor.obj",
        supportor_collision_path=None,
        scenario=None,
        scenario_probabilities=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        gear_stack_spacing=0.018,
        supportor_pose_base=(0.62, -0.10, 0.015),
        supportor_euler_base=(0.0, 0.0, 0.0),
        gear_to_support_pos=(0.0, 0.0, 0.31),
        gear_to_support_euler=(0.0, 0.0, 0.0),
        support_to_place_pos=(-0.20, 0.25, 0.045),
        support_to_place_euler=None,
        placed_gear_x_jit=0.10,
        placed_gear_y_jit=0.10,
        placed_gear_z_spacing=None,
        placed_gear_min_target_clearance=0.06,
        placed_gear_min_pair_clearance=0.08,
        placed_gear_max_sample_attempts=100,
        randomize_task_pose=True,
        task_x_jit=0.10,
        task_y_jit=0.15,
        task_z_jit=0.0,
        task_yaw_jit=math.pi / 18.0,
        safe_approach=0.08,
        lift_height=0.14,
        place_approach_height=0.10,
        retreat_height=0.10,
        randomize_lighting=True,
        randomize_outlscene=True,
        outlscene_xyz_jit=0.02,
        outlscene_eul_jit=0.01,
        randomize_plane_height=True,
        plane_height_jit=0.008,
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
        robot_texture_patterns=(
            "checkers",
            "gradient",
            "noise",
            "plain",
        ),
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
        randomize_distractors=True,
        distractor_root="/mnt/storage/GoogleScannedObjects",
        distractor_num_range=(1, 5),
        distractor_target_size_range=(0.06, 0.16),
        distractor_workspace=((0.25, 0.78), (-0.42, 0.42)),
        distractor_clearance=0.04,
        distractor_path_clearance=0.04,
        distractor_min_target_mask_pixels=1,
        gear_mass=0.85,
        gear_lateral_friction=0.75,
        supportor_lateral_friction=0.01,
        fix_gear_to_gripper=True,
        randomize_object_in_hand_pose=True,
        object_in_hand_x_jit=0.002,
        object_in_hand_y_jit=0.002,
        object_in_hand_z_jit=0.003,
        object_in_hand_roll_jit=0.0174533,
        object_in_hand_pitch_jit=0.0174533,
        object_in_hand_yaw_jit=0.0349066,
        object_in_hand_debug=False,
        disable_gear_gripper_collision=True,
        gear_gripper_collision_links=(
            8,
            9,
            10,
            pandaEndEffectorIndex,
        ),
        hold_nominal_gripper_width_after_attach=True,
    ):
        """Build a randomized exposed-gear extraction scene.

        ``gear_assets`` must contain lower, middle and upper assets in that
        order. Each item needs ``name``, ``visual_path``, ``collision_path``
        and ``grasp_path``. ``scenario`` may be ``"lower"``, ``"middle"`` or
        ``"upper"``; when omitted it is sampled from ``scenario_probabilities``.
        The target and all gears below it are loaded on the support stack. Gears
        above the target represent earlier extraction steps and are loaded near
        the place target as task-specific distractors. The target remains the
        topmost gear on the support, so the original state machine is unchanged.

        ``placed_gear_x_jit`` and ``placed_gear_y_jit`` define the support-local
        rectangular randomization range around ``support_to_place_pos`` for
        those previously extracted gears. ``placed_gear_z_spacing`` gives each
        additional placed gear a different height; by default it reuses
        ``gear_stack_spacing``. Clearances keep them away from the current place
        target and from one another.
        """
        if env_mesh_path is None:
            raise ValueError("env_mesh_path must be provided")
        if supportor_path is None:
            raise ValueError("supportor_path must be provided")

        if gear_assets is None or len(gear_assets) != 3:
            raise ValueError(
                "gear_assets must contain exactly three items ordered "
                "[lower, middle, upper]"
            )
        required_asset_keys = {
            "name", "visual_path", "collision_path", "grasp_path"
        }
        self.gear_assets = [dict(asset) for asset in gear_assets]
        for asset in self.gear_assets:
            missing = required_asset_keys.difference(asset)
            if missing:
                raise ValueError(
                    f"gear asset {asset!r} is missing {sorted(missing)}"
                )
        gear_names = [asset["name"] for asset in self.gear_assets]
        if len(set(gear_names)) != 3:
            raise ValueError("gear asset names must be unique")

        scenario_names = ("lower", "middle", "upper")
        if scenario is None:
            probabilities = np.asarray(
                scenario_probabilities, dtype=float
            )
            if probabilities.shape != (3,) or np.any(probabilities < 0.0):
                raise ValueError(
                    "scenario_probabilities must be three non-negative values"
                )
            probability_sum = float(probabilities.sum())
            if probability_sum <= 0.0:
                raise ValueError("scenario probabilities must sum to > 0")
            probabilities /= probability_sum
            target_level = int(
                self.sceneRng.choice(3, p=probabilities)
            )
            scenario = scenario_names[target_level]
        else:
            scenario = str(scenario).lower()
            if scenario not in scenario_names:
                raise ValueError(
                    "scenario must be one of: lower, middle, upper"
                )
            target_level = scenario_names.index(scenario)

        if float(gear_stack_spacing) < 0.0:
            raise ValueError("gear_stack_spacing must be non-negative")
        self.scenario = scenario
        self.target_level = target_level
        self.active_gear_assets = self.gear_assets[: target_level + 1]
        self.placed_gear_assets = self.gear_assets[target_level + 1:]
        target_asset = self.active_gear_assets[-1]
        self.target_gear_name = target_asset["name"]
        self.env_mesh_path = env_mesh_path
        self.gear_path = target_asset["visual_path"]
        self.gear_collision_path = target_asset["collision_path"]
        self.supportor_path = supportor_path
        self.initial_grasp_path = target_asset["grasp_path"]
        self.gear_stack_spacing = float(gear_stack_spacing)

        placed_gear_x_jit = float(placed_gear_x_jit)
        placed_gear_y_jit = float(placed_gear_y_jit)
        if placed_gear_z_spacing is None:
            placed_gear_z_spacing = self.gear_stack_spacing
        placed_gear_z_spacing = float(placed_gear_z_spacing)
        placed_gear_min_target_clearance = float(
            placed_gear_min_target_clearance
        )
        placed_gear_min_pair_clearance = float(
            placed_gear_min_pair_clearance
        )
        placed_gear_max_sample_attempts = int(
            placed_gear_max_sample_attempts
        )
        if placed_gear_x_jit < 0.0 or placed_gear_y_jit < 0.0:
            raise ValueError("placed-gear jitter must be non-negative")
        if (
            len(self.placed_gear_assets) > 1
            and placed_gear_z_spacing <= 0.0
        ):
            raise ValueError(
                "placed_gear_z_spacing must be positive when multiple gears "
                "are used as placed distractors"
            )
        if (
            placed_gear_min_target_clearance < 0.0
            or placed_gear_min_pair_clearance < 0.0
        ):
            raise ValueError("placed-gear clearances must be non-negative")
        if placed_gear_max_sample_attempts <= 0:
            raise ValueError(
                "placed_gear_max_sample_attempts must be positive"
            )

        self.safe_approach = float(safe_approach)
        self.lift_height = float(lift_height)
        self.place_approach_height = float(place_approach_height)
        self.retreat_height = float(retreat_height)

        self.remove_gear_grasp_constraint()
        self.fix_gear_to_gripper = bool(fix_gear_to_gripper)
        self.randomize_object_in_hand_pose = bool(
            randomize_object_in_hand_pose
        )
        self.disable_gear_gripper_collision = bool(
            disable_gear_gripper_collision
        )
        self.gear_gripper_collision_links = tuple(
            dict.fromkeys(
                int(link)
                for link in gear_gripper_collision_links
            )
        )
        self.gear_gripper_collision_disabled = False
        self.hold_nominal_gripper_width_after_attach = bool(
            hold_nominal_gripper_width_after_attach
        )
        self.nominal_grasp_finger_width = None
        self.object_in_hand_pos_jit = np.array(
            [
                object_in_hand_x_jit,
                object_in_hand_y_jit,
                object_in_hand_z_jit,
            ],
            dtype=float,
        )
        self.object_in_hand_eul_jit = np.array(
            [
                object_in_hand_roll_jit,
                object_in_hand_pitch_jit,
                object_in_hand_yaw_jit,
            ],
            dtype=float,
        )
        if np.any(self.object_in_hand_pos_jit < 0.0):
            raise ValueError(
                "object-in-hand position jitter must be non-negative"
            )
        if np.any(self.object_in_hand_eul_jit < 0.0):
            raise ValueError(
                "object-in-hand Euler jitter must be non-negative"
            )
        self.object_in_hand_debug = bool(object_in_hand_debug)
        self.grasp_gear_to_ee = None
        self.nominal_ee_to_gear = None
        self.randomized_ee_to_gear = None

        supportor_pose = np.asarray(
            supportor_pose_base,
            dtype=float,
        )
        supportor_orn = np.asarray(
            self.bullet_client.getQuaternionFromEuler(
                np.asarray(supportor_euler_base, dtype=float)
            ),
            dtype=float,
        )
        if randomize_task_pose:
            supportor_pose, supportor_orn = (
                self.supportorPoseDR.sample_SE3_randomization(
                    pos=supportor_pose,
                    orn=supportor_orn,
                    x_jitter_range=task_x_jit,
                    y_jitter_range=task_y_jit,
                    z_jitter_range=task_z_jit,
                    z_euler_jitter_range=task_yaw_jit,
                )
            )
        supportor_pose = np.asarray(supportor_pose, dtype=float)
        supportor_orn = np.asarray(supportor_orn, dtype=float)

        self.gear_to_support_pos = np.asarray(
            gear_to_support_pos,
            dtype=float,
        )
        self.gear_to_support_orn = np.asarray(
            self.bullet_client.getQuaternionFromEuler(
                np.asarray(gear_to_support_euler, dtype=float)
            ),
            dtype=float,
        )
        gear_poses = []
        gear_orns = []
        for level in range(len(self.active_gear_assets)):
            local_pos = self.gear_to_support_pos + np.array(
                [0.0, 0.0, level * self.gear_stack_spacing],
                dtype=float,
            )
            gear_pose, gear_orn = self.bullet_client.multiplyTransforms(
                supportor_pose.tolist(),
                supportor_orn.tolist(),
                local_pos.tolist(),
                self.gear_to_support_orn.tolist(),
            )
            gear_poses.append(np.asarray(gear_pose, dtype=float))
            gear_orns.append(np.asarray(gear_orn, dtype=float))
        gear_pose = gear_poses[-1]
        gear_orn = gear_orns[-1]
        self.transport_gear_z = float(
            gear_pose[2] + self.lift_height
        )

        place_pos, _ = self.bullet_client.multiplyTransforms(
            supportor_pose.tolist(),
            supportor_orn.tolist(),
            np.asarray(
                support_to_place_pos,
                dtype=float,
            ).tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )
        self.place_gear_pos = np.asarray(place_pos, dtype=float)
        if support_to_place_euler is None:
            self.place_gear_orn = gear_orn.copy()
        else:
            place_local_orn = (
                self.bullet_client.getQuaternionFromEuler(
                    np.asarray(
                        support_to_place_euler,
                        dtype=float,
                    )
                )
            )
            _, place_orn = (
                self.bullet_client.multiplyTransforms(
                    [0.0, 0.0, 0.0],
                    supportor_orn.tolist(),
                    [0.0, 0.0, 0.0],
                    place_local_orn,
                )
            )
            self.place_gear_orn = np.asarray(
                place_orn,
                dtype=float,
            )

        # Gears above the current target have already been extracted in a real
        # long-horizon rollout. Place them near (but not on) this trajectory's
        # target so they act as realistic, gear-shaped distractors.
        placed_local_positions = []
        placed_gear_poses = []
        placed_gear_orns = []
        place_local_pos = np.asarray(support_to_place_pos, dtype=float)
        for placed_idx, _asset in enumerate(self.placed_gear_assets):
            sampled_local_pos = None
            for _ in range(placed_gear_max_sample_attempts):
                candidate = place_local_pos.copy()
                candidate[0] += self.sceneRng.uniform(
                    -placed_gear_x_jit,
                    placed_gear_x_jit,
                )
                candidate[1] += self.sceneRng.uniform(
                    -placed_gear_y_jit,
                    placed_gear_y_jit,
                )
                candidate[2] += placed_idx * placed_gear_z_spacing
                target_clearance = float(
                    np.linalg.norm(candidate[:2] - place_local_pos[:2])
                )
                pair_clearance = min(
                    (
                        float(np.linalg.norm(candidate[:2] - other[:2]))
                        for other in placed_local_positions
                    ),
                    default=np.inf,
                )
                if (
                    target_clearance >= placed_gear_min_target_clearance
                    and pair_clearance >= placed_gear_min_pair_clearance
                ):
                    sampled_local_pos = candidate
                    break
            if sampled_local_pos is None:
                raise ValueError(
                    "failed to sample non-overlapping placed gears; increase "
                    "placed_gear_x_jit/placed_gear_y_jit, reduce the clearance "
                    "values, or increase placed_gear_max_sample_attempts"
                )

            placed_local_positions.append(sampled_local_pos)
            placed_pos, _ = self.bullet_client.multiplyTransforms(
                supportor_pose.tolist(),
                supportor_orn.tolist(),
                sampled_local_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
            )
            # self.place_gear_orn is already in the world frame. The position
            # should follow the support frame, while orientation should match
            # a gear produced by the same place action.
            placed_gear_poses.append(np.asarray(placed_pos, dtype=float))
            placed_gear_orns.append(self.place_gear_orn.copy())

        self.supportor_collision_path = (
            supportor_collision_path
            if supportor_collision_path is not None
            else coacd_convex_decomposition(self.supportor_path)
        )
        self.com_gears = [
            get_com(asset["visual_path"])
            for asset in self.active_gear_assets
        ]
        self.com_placed_gears = [
            get_com(asset["visual_path"])
            for asset in self.placed_gear_assets
        ]
        self.com_gear = self.com_gears[-1]
        self.com_supportor = get_com(self.supportor_path)
        self.initial_grasp_guess = load_initial_grasp_pose(
            self.initial_grasp_path
        )

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

        self.supportor_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.supportor_path,
            vhacd_mesh_file=self.supportor_collision_path,
            desired_mass=0.0,
            position=supportor_pose,
            baseOrientation=supportor_orn,
            center_of_mass=np.asarray(self.com_supportor),
            lateral_friction=float(
                supportor_lateral_friction
            ),
            spinning_friction=0.0,
        )
        self.supportor_obj_id = self.supportor_id

        self.gear_ids = []
        self.placed_gear_ids = []
        self.gear_ids_by_name = {}
        for asset, pose, orn, com in zip(
            self.active_gear_assets,
            gear_poses,
            gear_orns,
            self.com_gears,
        ):
            gear_body_id = load_models(
                self.bullet_client,
                visual_mesh_file=asset["visual_path"],
                vhacd_mesh_file=asset["collision_path"],
                desired_mass=float(gear_mass),
                position=pose,
                baseOrientation=orn,
                center_of_mass=np.asarray(com),
                lateral_friction=float(gear_lateral_friction),
                spinning_friction=0.0002,
            )
            self.gear_ids.append(gear_body_id)
            self.gear_ids_by_name[asset["name"]] = gear_body_id

        for asset, pose, orn, com in zip(
            self.placed_gear_assets,
            placed_gear_poses,
            placed_gear_orns,
            self.com_placed_gears,
        ):
            gear_body_id = load_models(
                self.bullet_client,
                visual_mesh_file=asset["visual_path"],
                vhacd_mesh_file=asset["collision_path"],
                desired_mass=float(gear_mass),
                position=pose,
                baseOrientation=orn,
                center_of_mass=np.asarray(com),
                lateral_friction=float(gear_lateral_friction),
                spinning_friction=0.0002,
            )
            self.placed_gear_ids.append(gear_body_id)
            self.gear_ids_by_name[asset["name"]] = gear_body_id

        # All legacy expert/state-machine methods operate on this alias. It is
        # deliberately bound to the exposed (last loaded) gear only.
        self.gear_id = self.gear_ids[-1]
        self.gear_obj_id = self.gear_id
        self.pick_up_obj_id = self.gear_id
        self.gear_gripper_collision_disabled = False

        for finger_link in (9, 10):
            self.bullet_client.changeDynamics(
                self.panda,
                finger_link,
                lateralFriction=1.2,
                spinningFriction=0.01,
                rollingFriction=0.001,
            )

        if randomize_object_color:
            self.object_color_cfg = {}
            for asset, body_id in zip(
                self.active_gear_assets + self.placed_gear_assets,
                self.gear_ids + self.placed_gear_ids,
            ):
                self.object_color_cfg[asset["name"]] = (
                    self.objectColorDR.sample_and_apply_object_color_randomization(
                        body_id=body_id,
                        mode=object_color_mode,
                        strength=object_color_strength,
                        recolor_palette=object_recolor_palette,
                        recolor_target_color=(
                            object_recolor_target_color
                        ),
                        specular_range=object_specular_range,
                        alpha=None,
                    )
                )
            self.object_color_cfg["supportor"] = (
                self.objectColorDR.sample_and_apply_object_color_randomization(
                    body_id=self.supportor_id,
                    mode=object_color_mode,
                    strength=object_color_strength,
                    recolor_palette=object_recolor_palette,
                    recolor_target_color=object_recolor_target_color,
                    specular_range=object_specular_range,
                    alpha=None,
                )
            )
        else:
            self.objectColorDR.reset()
            self.object_color_cfg = None

        self.waite_scene_stable()

        if randomize_distractors:
            self.distractorDR.sample_and_load_distractors(
                distractor_root=distractor_root,
                num_range=distractor_num_range,
                target_size_range=(
                    distractor_target_size_range
                ),
                workspace=distractor_workspace,
                clearance=distractor_clearance,
                path_clearance=distractor_path_clearance,
                min_target_mask_pixels=(
                    distractor_min_target_mask_pixels
                ),
                target_body_id=self.gear_id,
                robot_body_id=self.panda,
                robot_base_offset=self.offset,
                planned_waypoints=(
                    self.get_state_machine_ee_waypoints()
                ),
                render_agentview_fn=self.get_agentview_image,
                end_effector_index=pandaEndEffectorIndex,
                ik_lower_limits=ll,
                ik_upper_limits=ul,
                ik_joint_ranges=jr,
                get_current_arm_joints_fn=(
                    self.get_current_arm_joints
                ),
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

        self.initial_gear_pos, self.initial_gear_orn = (
            self.get_gear_pose()
        )
        self.initial_gear_poses = [
            tuple(
                np.asarray(value, dtype=float)
                for value in get_true_PositionAndOrientation(
                    self.bullet_client, body_id
                )
            )
            for body_id in self.gear_ids
        ]
        self.initial_placed_gear_poses = [
            tuple(
                np.asarray(value, dtype=float)
                for value in get_true_PositionAndOrientation(
                    self.bullet_client, body_id
                )
            )
            for body_id in self.placed_gear_ids
        ]
        (
            self.initial_supportor_pos,
            self.initial_supportor_orn,
        ) = self.get_supportor_pose()
        self.target_gripper = self.GRIPPER_OPEN
        self.set_gripper_state(self.target_gripper)

        self.state_idx = 0
        self.state = self.states[0]
        self.state_t = 0.0
        self.done = False
        self.prepare_state(self.state)

    def get_support_pin_pose(self):
        supportor_pos, supportor_orn = self.get_supportor_pose()
        pin_pos, pin_orn = self.bullet_client.multiplyTransforms(
            supportor_pos.tolist(),
            supportor_orn.tolist(),
            self.gear_to_support_pos.tolist(),
            self.gear_to_support_orn.tolist(),
        )
        return (
            np.asarray(pin_pos, dtype=float),
            np.asarray(pin_orn, dtype=float),
        )

    def get_place_gear_pose(self, above=False):
        if self.place_gear_pos is None:
            raise RuntimeError(
                "make_scene() must be called before place queries"
            )
        pos = self.place_gear_pos.copy()
        if above:
            pos[2] = max(
                pos[2] + self.place_approach_height,
                float(self.transport_gear_z),
            )
        return pos, self.place_gear_orn.copy()

    def get_state_machine_ee_waypoints(self):
        grasp_pos, grasp_orn = self.get_initial_guess_grasp()
        pregrasp_pos = grasp_pos + np.array(
            [0.0, -self.safe_approach, 0.0],
            dtype=float,
        )
        lifted_pos = grasp_pos + np.array(
            [0.0, 0.0, self.lift_height],
            dtype=float,
        )

        annotated_gear_to_ee = (
            self.get_gear_to_ee_transform(
                ee_pos=grasp_pos,
                ee_orn=grasp_orn,
            )
        )
        above_gear_pos, above_gear_orn = (
            self.get_place_gear_pose(above=True)
        )
        place_gear_pos, place_gear_orn = (
            self.get_place_gear_pose(above=False)
        )
        above_ee_pos, above_ee_orn = (
            self.get_ee_pose_for_gear_pose(
                above_gear_pos,
                above_gear_orn,
                annotated_gear_to_ee,
            )
        )
        place_ee_pos, place_ee_orn = (
            self.get_ee_pose_for_gear_pose(
                place_gear_pos,
                place_gear_orn,
                annotated_gear_to_ee,
            )
        )
        retreat_pos = place_ee_pos + np.array(
            [0.0, 0.0, self.retreat_height],
            dtype=float,
        )
        return [
            (
                self.home_ee_pos.copy(),
                self.home_ee_orn.copy(),
            ),
            (pregrasp_pos, grasp_orn.copy()),
            (grasp_pos.copy(), grasp_orn.copy()),
            (lifted_pos, grasp_orn.copy()),
            (above_ee_pos, above_ee_orn),
            (place_ee_pos, place_ee_orn),
            (retreat_pos, place_ee_orn),
        ]

    def waite_scene_stable(
        self,
        waite_steps=1000,
        vel_threshold=0.005,
    ):
        for _ in range(int(waite_steps)):
            self.bullet_client.stepSimulation()
            speeds = []
            for body_id in self.gear_ids + self.placed_gear_ids:
                linear_vel, angular_vel = (
                    self.bullet_client.getBaseVelocity(body_id)
                )
                speeds.append(
                    np.linalg.norm(linear_vel)
                    + np.linalg.norm(angular_vel)
                )
            if max(speeds, default=0.0) < float(vel_threshold):
                print("Scene stabilized.")
                return True
        print(
            "Warning: gear did not stabilize within timeout."
        )
        return False

    def get_initial_guess_grasp(self):
        gear_pos, gear_orn = self.get_gear_pose()
        grasp_pos, grasp_orn = (
            self.bullet_client.multiplyTransforms(
                gear_pos.tolist(),
                gear_orn.tolist(),
                self.initial_grasp_guess["t"],
                self.initial_grasp_guess["quat"],
            )
        )
        return (
            np.asarray(grasp_pos, dtype=float),
            np.asarray(grasp_orn, dtype=float),
        )

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
        """High-level binary gripper command: 1=open, 0=closed.

        After the gear has been rigidly attached, the policy/action label stays
        CLOSED (0), but the simulated finger motor target is held at the actual
        width reached by the preceding nominal physical grasp. This prevents the
        fingers from snapping shut when gear/gripper collision is disabled.
        """
        self.target_gripper = (
            self.GRIPPER_OPEN if float(state) >= 0.5 else self.GRIPPER_CLOSED
        )

        hold_width = getattr(self, "nominal_grasp_finger_width", None)
        should_hold_nominal_width = (
            self.target_gripper == self.GRIPPER_CLOSED
            and getattr(self, "hold_nominal_gripper_width_after_attach", False)
            and getattr(self, "gear_grasp_constraint_id", None) is not None
            and hold_width is not None
        )

        if should_hold_nominal_width:
            self.set_gripper_width(hold_width)
        else:
            self.set_gripper_width(
                self.gripper_state_to_width(self.target_gripper)
            )

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

    def prepare_state(self, state):
        ee_pos, ee_orn = self.get_ee_pose()
        self.motion_start_pos = ee_pos.copy()
        self.motion_start_orn = ee_orn.copy()
        self.motion_target_pos = ee_pos.copy()
        self.motion_target_orn = ee_orn.copy()
        self.gear_motion_start_pos = None
        self.gear_motion_start_orn = None
        self.gear_motion_target_pos = None
        self.gear_motion_target_orn = None

        if state == "home":
            self.motion_target_pos = self.home_ee_pos.copy()
            self.motion_target_orn = self.home_ee_orn.copy()
            self.target_gripper = self.GRIPPER_OPEN
            self.set_gripper_state(self.target_gripper)

        elif state == "move_pregrasp":
            grasp_pos, grasp_orn = (
                self.get_initial_guess_grasp()
            )
            self.motion_target_pos = grasp_pos + np.array(
                [0.0, -self.safe_approach, 0.0],
                dtype=float,
            )
            self.motion_target_orn = grasp_orn.copy()
            self.target_gripper = self.GRIPPER_OPEN

        elif state == "open_gripper":
            self.target_gripper = self.GRIPPER_OPEN

        elif state == "move_grasp":
            grasp_pos, grasp_orn = (
                self.get_initial_guess_grasp()
            )
            self.last_grasp_pose = grasp_pos.copy()
            self.last_grasp_orn = grasp_orn.copy()
            self.motion_target_pos = grasp_pos.copy()
            self.motion_target_orn = grasp_orn.copy()
            self.target_gripper = self.GRIPPER_OPEN

        elif state == "close_gripper":
            self.target_gripper = self.GRIPPER_CLOSED

        elif state == "lift_gear":
            self.motion_target_pos = ee_pos + np.array(
                [0.0, 0.0, self.lift_height],
                dtype=float,
            )
            self.motion_target_orn = ee_orn.copy()
            self.target_gripper = self.GRIPPER_CLOSED

        elif state in (
            "move_above_place",
            "lower_to_place",
        ):
            (
                self.gear_motion_start_pos,
                self.gear_motion_start_orn,
            ) = self.get_gear_pose()
            above = state == "move_above_place"
            (
                self.gear_motion_target_pos,
                self.gear_motion_target_orn,
            ) = self.get_place_gear_pose(above=above)
            self.target_gripper = self.GRIPPER_CLOSED

        elif state == "release_gear":
            self.motion_target_pos = ee_pos.copy()
            self.motion_target_orn = ee_orn.copy()
            self.target_gripper = self.GRIPPER_OPEN
            self.set_gripper_state(self.target_gripper)

        elif state == "retreat":
            self.motion_target_pos = ee_pos + np.array(
                [0.0, 0.0, self.retreat_height],
                dtype=float,
            )
            self.motion_target_orn = ee_orn.copy()
        else:
            raise ValueError(
                f"Unknown GearHorizonSim state: {state}"
            )

    def switch_to_next_state(self):
        if self.state == "close_gripper":
            self.randomize_and_fix_gear_to_gripper()
        # elif self.state == "lower_to_place":
        #     self.remove_gear_grasp_constraint()

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
        progress = min(
            self.state_t / max(duration, 1e-8),
            1.0,
        )

        if self.state in (
            "open_gripper",
            "close_gripper",
            "release_gear",
        ):
            self.set_gripper_state(self.target_gripper)
            self.target_pos, self.target_orn = (
                self.get_ee_pose()
            )

        elif self.state in (
            "move_above_place",
            "lower_to_place",
        ):
            desired_gear_pos = (
                (1.0 - progress)
                * self.gear_motion_start_pos
                + progress
                * self.gear_motion_target_pos
            )
            desired_gear_orn = quat_slerp(
                self.gear_motion_start_orn,
                self.gear_motion_target_orn,
                progress,
            )
            self.target_pos, self.target_orn = (
                self.get_ee_pose_for_gear_pose(
                    desired_gear_pos,
                    desired_gear_orn,
                )
            )
            self.solve_ik_and_apply(
                self.target_pos,
                self.target_orn,
            )
            self.set_gripper_state(self.target_gripper)

        else:
            self.target_pos = (
                (1.0 - progress)
                * self.motion_start_pos
                + progress
                * self.motion_target_pos
            )
            self.target_orn = quat_slerp(
                self.motion_start_orn,
                self.motion_target_orn,
                progress,
            )
            self.solve_ik_and_apply(
                self.target_pos,
                self.target_orn,
            )
            self.set_gripper_state(self.target_gripper)

        if self.state_t >= duration:
            self.switch_to_next_state()

        return self.target_pos, self.target_orn

    def collect_observation(self, use_agent_cam=True, direct=False,
                            collect_gear_ee=False, use_eye_in_hand=True):
        """Collect policy observations in the original Franka EE frame."""
        del collect_gear_ee
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
        """Return [EE xyz, EE quaternion, binary gripper] with no gear TCP conversion."""
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

    def get_gripper_mean_width(self):
        finger_widths = np.array(
            [
                self.bullet_client.getJointState(
                    self.panda,
                    9,
                )[0],
                self.bullet_client.getJointState(
                    self.panda,
                    10,
                )[0],
            ],
            dtype=float,
        )
        return float(np.mean(finger_widths))

    def is_gear_grasped(self):
        if self.gear_grasp_constraint_id is not None:
            return True
        contacts = self.bullet_client.getContactPoints(
            bodyA=self.panda,
            bodyB=self.gear_id,
        )
        return (
            len(contacts) > 0
            and self.get_gripper_mean_width() < 0.03
        )

    def get_extraction_metrics(self):
        if (
            not hasattr(self, "gear_id")
            or self.place_gear_pos is None
        ):
            return None

        gear_pos, gear_orn = self.get_gear_pose()
        pin_pos, _ = self.get_support_pin_pose()
        position_error = float(
            np.linalg.norm(
                gear_pos - self.place_gear_pos
            )
        )
        xy_position_error = float(
            np.linalg.norm(
                gear_pos[:2]
                - self.place_gear_pos[:2]
            )
        )
        height_error = float(
            abs(
                gear_pos[2]
                - self.place_gear_pos[2]
            )
        )
        support_xy_separation = float(
            np.linalg.norm(
                gear_pos[:2] - pin_pos[:2]
            )
        )
        orientation_error_rad = float(
            np.linalg.norm(
                (
                    R.from_quat(
                        self.place_gear_orn
                    ).inv()
                    * R.from_quat(gear_orn)
                ).as_rotvec()
            )
        )
        return {
            "position_error": position_error,
            "xy_position_error": xy_position_error,
            "height_error": height_error,
            "support_xy_separation": (
                support_xy_separation
            ),
            "orientation_error_rad": (
                orientation_error_rad
            ),
            "gear_pos": gear_pos,
            "target_pos": self.place_gear_pos.copy(),
            "pin_pos": pin_pos,
        }

    def is_success(
        self,
        position_tol=0.06,
        height_tol=0.1,
        min_support_xy_separation=0.05,
        require_released=False,
        require_done=False,
        return_info=False,
        debug=False,
    ):
        info = {
            "success": False,
            "reason": None,
        }

        def finish(value):
            info["success"] = bool(value)
            if debug:
                print(
                    "\n====== [GearHorizonSim is_success] "
                    "=========="
                )
                for key, val in info.items():
                    print(f"{key}: {val}")
                print(
                    "================================"
                    "==========\n"
                )
            if return_info:
                return bool(value), info
            return bool(value)

        if require_done and not self.done:
            info["reason"] = (
                "state_machine_not_done"
            )
            return finish(False)

        metrics = self.get_extraction_metrics()
        if metrics is None:
            info["reason"] = "missing_scene"
            return finish(False)

        info.update({
            "position_error": metrics[
                "position_error"
            ],
            "xy_position_error": metrics[
                "xy_position_error"
            ],
            "height_error": metrics[
                "height_error"
            ],
            "support_xy_separation": metrics[
                "support_xy_separation"
            ],
            "orientation_error_deg": math.degrees(
                metrics["orientation_error_rad"]
            ),
            "position_tol": float(position_tol),
            "height_tol": float(height_tol),
            "min_support_xy_separation": float(
                min_support_xy_separation
            ),
        })

        if (
            metrics["position_error"]
            > float(position_tol)
        ):
            info["reason"] = (
                "gear_not_at_place_target"
            )
            return finish(False)

        if (
            metrics["height_error"]
            > float(height_tol)
        ):
            info["reason"] = (
                "gear_not_near_place_height"
            )
            return finish(False)

        if (
            metrics["support_xy_separation"]
            < float(min_support_xy_separation)
        ):
            info["reason"] = (
                "gear_not_extracted_from_support"
            )
            return finish(False)

        if require_released:
            released = (
                self.gear_grasp_constraint_id
                is None
                and self.target_gripper
                == self.GRIPPER_OPEN
            )
            info["gear_released"] = bool(released)
            if not released:
                info["reason"] = (
                    "gear_not_released"
                )
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
