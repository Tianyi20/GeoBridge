import os
import glob
import numpy as np
from tqdm import tqdm


class DistractorDR(object):
    """
    Distractor domain randomization using Google Scanned Objects.

    The randomizer owns all distractor-specific state and logic:
      - GSO discovery
      - OBJ bbox parsing / caching
      - bbox collision body creation
      - random shape / number / pose sampling
      - target visibility mask check
      - final PyBullet collision / robot-plan clearance checks

    Task-specific information is passed in through arguments/callbacks so this class
    can be reused by other manipulation tasks or multiprocessing workers that each
    have their own bullet_client.
    """

    def __init__(self, bullet_client, seed=42):
        self.bullet_client = bullet_client
        self.distractor_rng = np.random.default_rng(seed)
        self.distractor_ids = []
        self.distractor_metadata = []
        self.gso_bbox_cache = {}

    def clear_distractors(self):
        for body_id in self.distractor_ids:
            try:
                self.bullet_client.removeBody(body_id)
            except Exception:
                pass
        self.distractor_ids = []
        self.distractor_metadata = []

    def discover_gso_meshes(self, distractor_root):
        pattern = os.path.join(distractor_root, "*", "1", "meshes", "model.obj")
        meshes = sorted(glob.glob(pattern))
        if len(meshes) == 0:
            print(f"Warning: no GSO model.obj found under {distractor_root}")
        return meshes

    def read_obj_bbox(self, obj_path):
        if obj_path in self.gso_bbox_cache:
            return self.gso_bbox_cache[obj_path]

        vertices = []
        try:
            with open(obj_path, "r", errors="ignore") as f:
                for line in f:
                    if not line.startswith("v "):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
        except OSError as exc:
            print(f"Warning: cannot read OBJ {obj_path}: {exc}")
            return None

        if len(vertices) == 0:
            print(f"Warning: no vertices found in OBJ {obj_path}")
            return None

        vertices = np.asarray(vertices, dtype=np.float32)
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        dims = vmax - vmin
        if (not np.all(np.isfinite(dims))) or float(np.max(dims)) <= 1e-8:
            print(f"Warning: invalid bbox for OBJ {obj_path}")
            return None

        center = 0.5 * (vmin + vmax)
        bbox = (vmin, vmax, center, dims)
        self.gso_bbox_cache[obj_path] = bbox
        return bbox

    @staticmethod
    def point_segment_distance_xy(point_xy, a_xy, b_xy):
        point_xy = np.asarray(point_xy, dtype=float)
        a_xy = np.asarray(a_xy, dtype=float)
        b_xy = np.asarray(b_xy, dtype=float)
        ab = b_xy - a_xy
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(point_xy - a_xy))
        t = float(np.clip(np.dot(point_xy - a_xy, ab) / denom, 0.0, 1.0))
        closest = a_xy + t * ab
        return float(np.linalg.norm(point_xy - closest))

    def is_distractor_xy_safe(self,
                              xy,
                              half_extents,
                              target_body_id,
                              planned_waypoints,
                              robot_base_offset,
                              clearance=0.035,
                              path_clearance=0.10):
        xy = np.asarray(xy, dtype=float)
        half_extents = np.asarray(half_extents, dtype=float)
        robot_base_offset = np.asarray(robot_base_offset, dtype=float)
        radius_xy = float(np.linalg.norm(half_extents[:2])) + clearance

        try:
            aabb_min, aabb_max = self.bullet_client.getAABB(target_body_id)
            aabb_min = np.asarray(aabb_min, dtype=float)
            aabb_max = np.asarray(aabb_max, dtype=float)
            obj_center_xy = 0.5 * (aabb_min[:2] + aabb_max[:2])
            obj_radius_xy = 0.5 * float(np.linalg.norm(aabb_max[:2] - aabb_min[:2]))
            if np.linalg.norm(xy - obj_center_xy) < obj_radius_xy + radius_xy + clearance:
                return False
        except Exception:
            pass

        if np.linalg.norm(xy - robot_base_offset[:2]) < 0.22 + radius_xy:
            return False

        for i in range(len(planned_waypoints) - 1):
            a = planned_waypoints[i][0][:2]
            b = planned_waypoints[i + 1][0][:2]
            dist = self.point_segment_distance_xy(xy, a, b)
            if dist < path_clearance + radius_xy:
                return False

        for meta in self.distractor_metadata:
            other_xy = np.asarray(meta["xy"], dtype=float)
            other_r = float(meta["radius_xy"])
            if np.linalg.norm(xy - other_xy) < radius_xy + other_r + clearance:
                return False

        return True

    def create_bbox_collision_gso_body(self, obj_path, xy, yaw, target_size):
        bbox = self.read_obj_bbox(obj_path)
        if bbox is None:
            return None, None

        _, _, bbox_center, dims = bbox
        max_dim = float(np.max(dims))
        if max_dim <= 1e-8:
            return None, None

        scale = float(target_size) / max_dim
        half_extents = 0.5 * np.asarray(dims, dtype=float) * scale
        half_extents = np.maximum(half_extents, np.array([0.008, 0.008, 0.008], dtype=float))
        visual_shift = (-np.asarray(bbox_center, dtype=float) * scale).tolist()

        base_pos = [float(xy[0]), float(xy[1]), float(half_extents[2]) + 0.003]
        base_orn = self.bullet_client.getQuaternionFromEuler([0.0, 0.0, float(yaw)])

        visual_shape = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_MESH,
            fileName=obj_path,
            meshScale=[scale, scale, scale],
            visualFramePosition=visual_shift,
        )
        collision_shape = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=half_extents.tolist(),
        )

        body_id = self.bullet_client.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=base_pos,
            baseOrientation=base_orn,
        )

        texture_path = os.path.join(os.path.dirname(obj_path), "texture.png")
        if os.path.exists(texture_path):
            try:
                texture_id = self.bullet_client.loadTexture(texture_path)
                self.bullet_client.changeVisualShape(body_id, -1, textureUniqueId=texture_id)
            except Exception:
                pass

        self.bullet_client.changeDynamics(
            body_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.01,
            rollingFriction=0.01,
            restitution=0.0,
        )

        meta = {
            "body_id": body_id,
            "obj_path": obj_path,
            "xy": [float(xy[0]), float(xy[1])],
            "yaw": float(yaw),
            "scale": float(scale),
            "target_size": float(target_size),
            "half_extents": half_extents.tolist(),
            "radius_xy": float(np.linalg.norm(half_extents[:2])),
        }
        return body_id, meta

    def get_all_robot_joint_positions(self, robot_body_id):
        return [self.bullet_client.getJointState(robot_body_id, j)[0]
                for j in range(self.bullet_client.getNumJoints(robot_body_id))]

    def restore_all_robot_joint_positions(self, robot_body_id, q_all):
        for j, q in enumerate(q_all):
            self.bullet_client.resetJointState(robot_body_id, j, q)

    def distractor_too_close_to_robot_plan(self,
                                           body_id,
                                           robot_body_id,
                                           end_effector_index,
                                           planned_waypoints,
                                           ik_lower_limits,
                                           ik_upper_limits,
                                           ik_joint_ranges,
                                           get_current_arm_joints_fn,
                                           quat_slerp_fn,
                                           panda_num_dofs=7,
                                           clearance=0.035,
                                           samples_per_segment=5):
        q_saved = self.get_all_robot_joint_positions(robot_body_id)
        try:
            for i in range(len(planned_waypoints) - 1):
                p0, q0 = planned_waypoints[i]
                p1, q1 = planned_waypoints[i + 1]
                for s in np.linspace(0.0, 1.0, samples_per_segment):
                    pos = (1.0 - s) * p0 + s * p1
                    orn = quat_slerp_fn(q0, q1, float(s))
                    ik = self.bullet_client.calculateInverseKinematics(
                        robot_body_id,
                        end_effector_index,
                        pos.tolist(),
                        orn.tolist(),
                        ik_lower_limits,
                        ik_upper_limits,
                        ik_joint_ranges,
                        get_current_arm_joints_fn(),
                        maxNumIterations=80,
                    )
                    for j in range(panda_num_dofs):
                        self.bullet_client.resetJointState(robot_body_id, j, ik[j])

                    for finger_opening in [0.04, 0.0]:
                        # Franka Panda finger joints; other tasks can ignore this by passing a robot
                        # without these joint indices only if they override/extend this method.
                        if self.bullet_client.getNumJoints(robot_body_id) > 10:
                            self.bullet_client.resetJointState(robot_body_id, 9, finger_opening)
                            self.bullet_client.resetJointState(robot_body_id, 10, finger_opening)
                        self.bullet_client.performCollisionDetection()
                        pts = self.bullet_client.getClosestPoints(
                            bodyA=robot_body_id,
                            bodyB=body_id,
                            distance=clearance,
                        )
                        if len(pts) > 0:
                            return True
            return False
        finally:
            self.restore_all_robot_joint_positions(robot_body_id, q_saved)
            self.bullet_client.performCollisionDetection()

    @staticmethod
    def body_mask_pixel_count(render_agentview_fn, body_id):
        _, _, _, _, seg = render_agentview_fn()
        body_uid = np.bitwise_and(seg, (1 << 24) - 1)
        return int(np.count_nonzero(body_uid == int(body_id)))

    @staticmethod
    def target_mask_is_visible(render_agentview_fn, target_body_id, min_target_mask_pixels=10):
        _, _, _, _, seg = render_agentview_fn()
        body_uid = np.bitwise_and(seg, (1 << 24) - 1)
        current_pixels = int(np.count_nonzero(body_uid == int(target_body_id)))
        return current_pixels >= int(max(1, min_target_mask_pixels))

    def loaded_distractor_is_safe(self,
                                  body_id,
                                  target_body_id,
                                  robot_body_id,
                                  end_effector_index,
                                  planned_waypoints,
                                  ik_lower_limits,
                                  ik_upper_limits,
                                  ik_joint_ranges,
                                  get_current_arm_joints_fn,
                                  quat_slerp_fn,
                                  render_agentview_fn,
                                  panda_num_dofs=7,
                                  clearance=0.035,
                                  min_target_mask_pixels=10,
                                  ):
        self.bullet_client.performCollisionDetection()

        pts = self.bullet_client.getClosestPoints(
            bodyA=body_id,
            bodyB=target_body_id,
            distance=clearance,
        )
        if len(pts) > 0:
            return False

        for other_id in self.distractor_ids:
            pts = self.bullet_client.getClosestPoints(
                bodyA=body_id,
                bodyB=other_id,
                distance=clearance,
            )
            if len(pts) > 0:
                return False

        if self.distractor_too_close_to_robot_plan(
            body_id=body_id,
            robot_body_id=robot_body_id,
            end_effector_index=end_effector_index,
            planned_waypoints=planned_waypoints,
            ik_lower_limits=ik_lower_limits,
            ik_upper_limits=ik_upper_limits,
            ik_joint_ranges=ik_joint_ranges,
            get_current_arm_joints_fn=get_current_arm_joints_fn,
            quat_slerp_fn=quat_slerp_fn,
            panda_num_dofs=panda_num_dofs,
            clearance=clearance,
        ):
            return False

        if not self.target_mask_is_visible(
            render_agentview_fn=render_agentview_fn,
            target_body_id=target_body_id,
            min_target_mask_pixels=min_target_mask_pixels,
        ):
            return False

        return True

    def sample_and_load_distractors(self,
                                    distractor_root="/mnt/storage/GoogleScannedObjects",
                                    num_range=(1, 5),
                                    target_size_range=(0.06, 0.16),
                                    workspace=((0.25, 0.78), (-0.42, 0.42)),
                                    clearance=0.035,
                                    path_clearance=0.10,
                                    min_target_mask_pixels=10,
                                    max_attempts_per_distractor=80,
                                    target_body_id=None,
                                    robot_body_id=None,
                                    robot_base_offset=None,
                                    planned_waypoints=None,
                                    render_agentview_fn=None,
                                    end_effector_index=None,
                                    ik_lower_limits=None,
                                    ik_upper_limits=None,
                                    ik_joint_ranges=None,
                                    get_current_arm_joints_fn=None,
                                    quat_slerp_fn=None,
                                    panda_num_dofs=7):
        self.clear_distractors()

        required = {
            "target_body_id": target_body_id,
            "robot_body_id": robot_body_id,
            "robot_base_offset": robot_base_offset,
            "planned_waypoints": planned_waypoints,
            "render_agentview_fn": render_agentview_fn,
            "end_effector_index": end_effector_index,
            "ik_lower_limits": ik_lower_limits,
            "ik_upper_limits": ik_upper_limits,
            "ik_joint_ranges": ik_joint_ranges,
            "get_current_arm_joints_fn": get_current_arm_joints_fn,
            "quat_slerp_fn": quat_slerp_fn,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"Missing required distractorDR arguments: {missing}")

        meshes = self.discover_gso_meshes(distractor_root)
        if len(meshes) == 0:
            return []

        rng = self.distractor_rng
        low, high = int(num_range[0]), int(num_range[1])
        if high < low:
            low, high = high, low
        num_distractors = int(rng.integers(low, high + 1))

        min_size, max_size = float(target_size_range[0]), float(target_size_range[1])
        if max_size < min_size:
            min_size, max_size = max_size, min_size

        x_bounds, y_bounds = workspace

        for k in tqdm(range(num_distractors), desc="Loading distractors"):
            accepted = False
            for attempt in range(max_attempts_per_distractor):
                obj_path = str(rng.choice(meshes))
                bbox = self.read_obj_bbox(obj_path)
                if bbox is None:
                    continue

                _, _, _, dims = bbox
                target_size = float(rng.uniform(min_size, max_size))
                scale = target_size / float(np.max(dims))
                half_extents = 0.5 * np.asarray(dims, dtype=float) * scale
                half_extents = np.maximum(half_extents, np.array([0.008, 0.008, 0.008], dtype=float))

                xy = np.array([
                    rng.uniform(float(x_bounds[0]), float(x_bounds[1])),
                    rng.uniform(float(y_bounds[0]), float(y_bounds[1])),
                ], dtype=float)
                yaw = float(rng.uniform(-np.pi, np.pi))

                if not self.is_distractor_xy_safe(
                    xy=xy,
                    half_extents=half_extents,
                    target_body_id=target_body_id,
                    planned_waypoints=planned_waypoints,
                    robot_base_offset=robot_base_offset,
                    clearance=clearance,
                    path_clearance=path_clearance,
                ):
                    continue

                body_id, meta = self.create_bbox_collision_gso_body(
                    obj_path=obj_path,
                    xy=xy,
                    yaw=yaw,
                    target_size=target_size,
                )
                if body_id is None:
                    continue

                if self.loaded_distractor_is_safe(
                    body_id=body_id,
                    target_body_id=target_body_id,
                    robot_body_id=robot_body_id,
                    end_effector_index=end_effector_index,
                    planned_waypoints=planned_waypoints,
                    ik_lower_limits=ik_lower_limits,
                    ik_upper_limits=ik_upper_limits,
                    ik_joint_ranges=ik_joint_ranges,
                    get_current_arm_joints_fn=get_current_arm_joints_fn,
                    quat_slerp_fn=quat_slerp_fn,
                    render_agentview_fn=render_agentview_fn,
                    panda_num_dofs=panda_num_dofs,
                    clearance=clearance,
                    min_target_mask_pixels=min_target_mask_pixels,
                ):
                    self.distractor_ids.append(body_id)
                    self.distractor_metadata.append(meta)
                    accepted = True
                    break

                self.bullet_client.removeBody(body_id)

            if not accepted:
                print(
                    f"Warning: failed to place distractor {k + 1}/{num_distractors} "
                    f"after {max_attempts_per_distractor} attempts."
                )

        print(f"Loaded {len(self.distractor_ids)}/{num_distractors} GSO distractors.")
        return self.distractor_ids
