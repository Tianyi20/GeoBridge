import numpy as np
import igl
import trimesh
import slippage_reshaping as sr
from pathlib import Path
import open3d as o3d
from ultility import load_initial_grasp_pose
from icecream import ic


class ShapeAugmentor:
    def __init__(self, obj_path, initial_grasp_path=None, auto_repair=True):
        self.obj_path = Path(obj_path)

        self.mesh = trimesh.load(
            self.obj_path,
            process=False,
            maintain_order=True,
        )

        self.initial_grasp_guess = (
            load_initial_grasp_pose(initial_grasp_path)
            if initial_grasp_path is not None
            else None
        )

        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a single Trimesh, got {type(self.mesh)}")

        self.V_opt = None
        self.face_k1 = None
        self.face_k2 = None
        self._vertex_adj = None

        self._sync_arrays()

        if auto_repair and not self.is_manifold():
            print("Input mesh is not manifold. Running safe in-place repair...")
            self.repair_mesh(inplace=True)

    # ============================================================
    # Mesh utilities
    # ============================================================

    def _sync_arrays(self):
        self.V = np.asarray(self.mesh.vertices, dtype=np.float64)
        self.F = np.asarray(self.mesh.faces, dtype=np.int64)

        # Any mesh topology/face update invalidates cached adjacency.
        self._vertex_adj = None

        if self.F.ndim != 2 or self.F.shape[1] != 3:
            raise ValueError(f"Input mesh must be triangulated, got F.shape={self.F.shape}")

    def _to_o3d(self):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.faces))
        return mesh_o3d

    def mesh_status(self):
        mesh_o3d = self._to_o3d()

        return {
            "edge_manifold_allow_boundary": mesh_o3d.is_edge_manifold(allow_boundary_edges=True),
            "edge_manifold_no_boundary": mesh_o3d.is_edge_manifold(allow_boundary_edges=False),
            "vertex_manifold": mesh_o3d.is_vertex_manifold(),
            "watertight": mesh_o3d.is_watertight(),
            "orientable": mesh_o3d.is_orientable(),
        }

    def is_manifold(self):
        status = self.mesh_status()
        return (
            status["edge_manifold_allow_boundary"]
            and status["vertex_manifold"]
            and status["orientable"]
        )

    def repair_mesh(self, inplace=True, merge_vertices=False):
        mesh = self.mesh if inplace else self.mesh.copy()

        before = self.mesh_status()

        # Safe repair:
        # 1. remove duplicated faces
        # 2. remove degenerate faces
        # 3. fix normals
        #
        # Do not remove unreferenced vertices or merge vertices by default,
        # because those operations can change vertex ids and break constraints/anchors.
        face_mask = mesh.unique_faces() & mesh.nondegenerate_faces()
        mesh.update_faces(face_mask)

        if merge_vertices:
            # Warning: this changes vertex ids.
            mesh.remove_unreferenced_vertices()
            mesh.merge_vertices(merge_tex=False, merge_norm=False)

        mesh.fix_normals()

        if inplace:
            self.mesh = mesh
            self._sync_arrays()
            self.face_k1 = None
            self.face_k2 = None
            self.V_opt = None

        after = self.mesh_status()

        print("Mesh repair status:")
        print("  before:", before)
        print("  after: ", after)

        if not self.is_manifold():
            print(
                "Warning: mesh is still not manifold after safe repair. "
                "A stronger repair would likely change topology, vertex ids, or texture seams."
            )

        return mesh

    # ============================================================
    # Slippage-preserving reshaping
    # ============================================================

    def get_fk(self, radius=5, use_k_ring=True):
        result = igl.principal_curvature(
            self.V,
            self.F,
            radius,
            use_k_ring,
        )

        _, _, PV1, PV2 = result[:4]

        PV1 = np.asarray(PV1, dtype=np.float64).reshape(-1)
        PV2 = np.asarray(PV2, dtype=np.float64).reshape(-1)

        if len(result) >= 5:
            bad_vertices = np.asarray(result[4], dtype=np.int64)
            if bad_vertices.size > 0:
                print(f"Warning: principal_curvature reported {bad_vertices.size} bad vertices")
                PV1[bad_vertices] = 0.0
                PV2[bad_vertices] = 0.0

        PV1 = np.nan_to_num(PV1, nan=0.0, posinf=0.0, neginf=0.0)
        PV2 = np.nan_to_num(PV2, nan=0.0, posinf=0.0, neginf=0.0)

        self.face_k1 = np.asarray(igl.average_onto_faces(self.F, PV1), dtype=np.float64)
        self.face_k2 = np.asarray(igl.average_onto_faces(self.F, PV2), dtype=np.float64)

        return self.face_k1, self.face_k2

    def slippage_reshape(
        self,
        constraint_ids,
        target_positions=None,
        max_iters=20,
        handle_error_distrib_enabled=False,
        input_name=None,
    ):
        constraint_ids = list(constraint_ids)

        if target_positions is None:
            raise ValueError("target_positions must be provided")

        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.shape != (len(constraint_ids), 3):
            raise ValueError(
                f"target_positions must have shape {(len(constraint_ids), 3)}, "
                f"got {target_positions.shape}"
            )

        # ------------------------------------------------------------
        # bbox_diag conversion layer
        #
        # External API convention:
        #   - target_positions is absolute target position in current mesh scale.
        #   - if called from displacement_reshape(), displacements are also absolute-scale
        #     because displacement_reshape() has already done:
        #         target = original + absolute_disp
        #
        # Internal C++ .deform convention:
        #   - displacement is normalized by bbox diagonal:
        #         normalized_disp = absolute_disp / bbox_diag
        #
        # This block makes the conversion explicit while keeping the public Python
        # interface absolute-scale.
        # ------------------------------------------------------------
        constraint_vertices = self.V[constraint_ids].copy()

        bbox_min = self.V.min(axis=0)
        bbox_max = self.V.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)

        if bbox_diag <= 0.0:
            raise ValueError(f"Invalid mesh bbox_diag={bbox_diag}")

        absolute_displacements = target_positions - constraint_vertices

        # This is what the optimizer should receive: absolute target positions.
        target_positions_for_solver = (
            constraint_vertices + absolute_displacements * bbox_diag
        )

        face_k1, face_k2 = self.get_fk()

        ic(constraint_ids)
        ic(target_positions_for_solver)

        V_opt = sr.optimize_mesh(
            self.V,
            self.F,
            face_k1,
            face_k2,
            constraint_ids,
            target_positions_for_solver,
            max_iters=max_iters,
            handle_error_distrib_enabled=handle_error_distrib_enabled,
            input_name=input_name or self.obj_path.stem,
        )

        self.V_opt = np.asarray(V_opt, dtype=np.float64)
        return self.V_opt

    def displacement_reshape(
        self,
        constraint_ids,
        displace_idxs,
        displacements,
        max_iters=20,
        handle_error_distrib_enabled=False,
        input_name=None,
    ):
        constraint_ids = list(constraint_ids)
        displace_idxs = list(displace_idxs)
        displacements = np.asarray(displacements, dtype=np.float64)

        if displacements.ndim == 1:
            if len(displace_idxs) != 1 or displacements.shape[0] != 3:
                raise ValueError(
                    "A 1D displacement must be a single 3D vector and requires exactly one displace_idx"
                )
            displacements = displacements.reshape(1, 3)

        if displacements.shape != (len(displace_idxs), 3):
            raise ValueError(
                f"displacements must have shape {(len(displace_idxs), 3)}, "
                f"got {displacements.shape}"
            )

        target_positions = self.V[constraint_ids].copy()
        ic(target_positions)

        for vid, disp in zip(displace_idxs, displacements):
            if vid not in constraint_ids:
                raise ValueError(f"displace_idxs must be a subset of constraint_ids, got {vid}")

            row = constraint_ids.index(vid)
            target_positions[row] += disp


        return self.slippage_reshape(
            constraint_ids=constraint_ids,
            target_positions=target_positions,
            max_iters=max_iters,
            handle_error_distrib_enabled=handle_error_distrib_enabled,
            input_name=input_name,
        )

    def write_augment_obj(self, output_path):
        if self.V_opt is None:
            raise ValueError("No optimized vertices found. Please run slippage_reshape() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        out_mesh = self.mesh.copy()
        out_mesh.vertices = np.asarray(self.V_opt, dtype=np.float64)
        out_mesh.export(output_path)

    # ============================================================
    # Geometry helpers for anchors and patches
    # ============================================================

    @staticmethod
    def _normalize(v, eps=1e-12):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < eps:
            return v
        return v / n

    @staticmethod
    def _point_triangle_barycentric(p, a, b, c, eps=1e-12):
        """
        Compute barycentric coordinates of point p w.r.t. triangle (a, b, c).
        Returns bary = [u, v, w], where p ~= u*a + v*b + w*c.
        """
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)

        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < eps:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        bary = np.array([u, v, w], dtype=np.float64)

        # closest_points from igl should already be inside/on the triangle, but this
        # avoids tiny numerical negatives.
        bary = np.clip(bary, 0.0, 1.0)
        s = bary.sum()
        if s < eps:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        return bary / s

    @staticmethod
    def _barycentric_point(V, F, face_id, bary):
        tri = F[int(face_id)]
        b0, b1, b2 = np.asarray(bary, dtype=np.float64)
        return b0 * V[tri[0]] + b1 * V[tri[1]] + b2 * V[tri[2]]

    def _build_vertex_adjacency(self):
        adj = [set() for _ in range(len(self.V))]
        for tri in self.F:
            i, j, k = map(int, tri)
            adj[i].update([j, k])
            adj[j].update([i, k])
            adj[k].update([i, j])
        return adj

    def get_k_ring_vertices(self, seed_vertex_ids, k_ring=2):
        """
        Return k-ring vertex ids around seed vertices.
        k_ring=0 returns only seed vertices.
        """
        seed_vertex_ids = [int(v) for v in seed_vertex_ids]

        if self._vertex_adj is None:
            self._vertex_adj = self._build_vertex_adjacency()

        visited = set(seed_vertex_ids)
        frontier = set(seed_vertex_ids)

        for _ in range(int(k_ring)):
            new_frontier = set()
            for v in frontier:
                new_frontier.update(self._vertex_adj[v])
            new_frontier -= visited
            visited.update(new_frontier)
            frontier = new_frontier

        return np.array(sorted(visited), dtype=np.int64)

    def closest_surface_anchor(self, point, k_ring=2):
        """
        Project a 3D point to the original mesh surface and create an anchor.

        Returns:
            anchor = {
                "face_id": int,
                "barycentric": [b0, b1, b2],
                "anchor_point_old": [x, y, z],
                "patch_vertex_ids": [...],
                "source_point_old": [x, y, z],
                "k_ring": int,
            }
        """
        point = np.asarray(point, dtype=np.float64).reshape(1, 3)

        sqrD, face_ids, closest_points = igl.point_mesh_squared_distance(
            point,
            self.V,
            self.F,
        )

        face_id = int(face_ids[0])
        anchor_point_old = np.asarray(closest_points[0], dtype=np.float64)

        tri = self.F[face_id]
        a, b, c = self.V[tri[0]], self.V[tri[1]], self.V[tri[2]]
        bary = self._point_triangle_barycentric(anchor_point_old, a, b, c)

        patch_vertex_ids = self.get_k_ring_vertices(seed_vertex_ids=tri, k_ring=k_ring)

        return {
            "face_id": face_id,
            "barycentric": bary.tolist(),
            "anchor_point_old": anchor_point_old.tolist(),
            "patch_vertex_ids": patch_vertex_ids.tolist(),
            "source_point_old": point.reshape(3).tolist(),
            "k_ring": int(k_ring),
            "squared_distance_to_mesh": float(sqrD[0]),
        }

    def _patch_weights_from_anchor(self, patch_ids, anchor_point_old, sigma=None):
        """
        Distance weights for local shape matching.
        Vertices closer to the anchor receive larger weights.
        """
        patch_ids = np.asarray(patch_ids, dtype=np.int64)
        P = self.V[patch_ids]
        anchor_point_old = np.asarray(anchor_point_old, dtype=np.float64).reshape(3)

        d = np.linalg.norm(P - anchor_point_old[None, :], axis=1)

        if sigma is None:
            sigma = np.median(d) + 1e-12

        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        return w + 1e-8

    # ============================================================
    # Local shape matching / Kabsch rigid fit
    # ============================================================

    @staticmethod
    def fit_rigid_transform(P, Q, weights=None):
        """
        Fit local rigid transform by Kabsch / Procrustes.

        Mathematical convention:
            q_i ~= R @ p_i + t

        Numpy row-vector convention:
            Q_hat = P @ R.T + t

        Args:
            P: old patch vertices, shape (K, 3)
            Q: new patch vertices, shape (K, 3)
            weights: optional, shape (K,)

        Returns:
            R: shape (3, 3), in SO(3)
            t: shape (3,)
        """
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        if P.shape != Q.shape:
            raise ValueError(f"P and Q must have same shape, got {P.shape} and {Q.shape}")
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"P and Q must be Kx3, got {P.shape}")
        if P.shape[0] < 3:
            raise ValueError("Need at least 3 points to fit a stable rigid transform")

        K = P.shape[0]

        if weights is None:
            weights = np.ones(K, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if weights.shape[0] != K:
                raise ValueError(f"weights must have shape ({K},), got {weights.shape}")

        weights = weights / (weights.sum() + 1e-12)

        p_center = np.sum(P * weights[:, None], axis=0)
        q_center = np.sum(Q * weights[:, None], axis=0)

        P0 = P - p_center
        Q0 = Q - q_center

        H = P0.T @ (Q0 * weights[:, None])

        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Avoid reflection.
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = q_center - R @ p_center
        return R, t

    # ============================================================
    # SE(3) helpers
    # ============================================================

    @staticmethod
    def _project_to_SO3(R):
        """
        Project a near-rotation matrix to SO(3).
        Useful if loaded R has small numerical errors.
        """
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")

        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt

        if np.linalg.det(R_proj) < 0:
            U[:, -1] *= -1.0
            R_proj = U @ Vt

        return R_proj

    @staticmethod
    def _quat_to_R(quat, order="xyzw"):
        """
        Convert quaternion to rotation matrix.

        order:
            "xyzw": [x, y, z, w]
            "wxyz": [w, x, y, z]
        """
        q = np.asarray(quat, dtype=np.float64).reshape(4)

        if order == "xyzw":
            x, y, z, w = q
        elif order == "wxyz":
            w, x, y, z = q
        else:
            raise ValueError("quat order must be 'xyzw' or 'wxyz'")

        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-12:
            raise ValueError("Quaternion norm is too small")

        w, x, y, z = w / n, x / n, y / n, z / n

        R = np.array([
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

        return R

    @staticmethod
    def _R_to_quat(R, order="xyzw"):
        """
        Convert rotation matrix to quaternion.
        Returns xyzw by default.
        """
        R = ShapeAugmentor._project_to_SO3(R)
        tr = np.trace(R)

        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        q_xyzw = np.array([x, y, z, w], dtype=np.float64)
        q_xyzw = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-12)

        if order == "xyzw":
            return q_xyzw
        if order == "wxyz":
            return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        raise ValueError("quat order must be 'xyzw' or 'wxyz'")

    def grasp_guess_to_SE3(self, grasp_guess=None, quat_order="xyzw"):
        """
        Convert loaded grasp guess to a 4x4 SE(3).

        Supported formats:
            1. 4x4 ndarray
            2. dict["T_mesh_hand_tcp"] : 4x4
            3. dict["T"]              : 4x4
            4. dict["R"] + dict["t"]
            5. dict["quat"] + dict["t"]

        Convention:
            T_mesh_hand_tcp means hand_tcp pose expressed in mesh frame.
            T[:3, 3] is TCP position in mesh frame.
            T[:3, :3] is TCP orientation in mesh frame.
        """
        if grasp_guess is None:
            grasp_guess = self.initial_grasp_guess

        if grasp_guess is None:
            raise ValueError("No grasp guess provided and self.initial_grasp_guess is None")

        if isinstance(grasp_guess, np.ndarray):
            T = np.asarray(grasp_guess, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"grasp_guess ndarray must be 4x4, got {T.shape}")
            T = T.copy()
            T[:3, :3] = self._project_to_SO3(T[:3, :3])
            T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
            return T

        if not isinstance(grasp_guess, dict):
            raise TypeError(f"grasp_guess must be dict or 4x4 ndarray, got {type(grasp_guess)}")

        if "T_mesh_hand_tcp" in grasp_guess:
            T = np.asarray(grasp_guess["T_mesh_hand_tcp"], dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"T_mesh_hand_tcp must be 4x4, got {T.shape}")
            T = T.copy()
            T[:3, :3] = self._project_to_SO3(T[:3, :3])
            T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
            return T

        if "T" in grasp_guess:
            T = np.asarray(grasp_guess["T"], dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"T must be 4x4, got {T.shape}")
            T = T.copy()
            T[:3, :3] = self._project_to_SO3(T[:3, :3])
            T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
            return T

        if "t" not in grasp_guess:
            raise KeyError("grasp_guess must contain 't' if no 4x4 transform is provided")

        t = np.asarray(grasp_guess["t"], dtype=np.float64).reshape(3)

        if "R" in grasp_guess:
            R = np.asarray(grasp_guess["R"], dtype=np.float64)
            R = self._project_to_SO3(R)
        elif "quat" in grasp_guess:
            R = self._quat_to_R(grasp_guess["quat"], order=quat_order)
        else:
            raise KeyError("grasp_guess must contain either 'R' or 'quat' if no 4x4 transform is provided")

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    # ============================================================
    # SE(3) grasp transfer
    # ============================================================

    def make_grasp_anchor_from_SE3(self, T_grasp_old=None, k_ring=2, quat_order="xyzw"):
        """
        Build and return the surface anchor for an old SE(3) grasp pose.

        By default, the anchor is the closest surface point to the old TCP
        translation T[:3, 3]. For more accurate grasping, you can later replace
        this with real finger contact anchors.
        """
        T_grasp_old = self.grasp_guess_to_SE3(T_grasp_old, quat_order=quat_order)
        tcp_old = T_grasp_old[:3, 3]
        return self.closest_surface_anchor(point=tcp_old, k_ring=k_ring)

    def transfer_grasp_SE3_by_anchor(
        self,
        T_grasp_old,
        anchor,
        use_distance_weights=True,
        quat_order="xyzw",
    ):
        """
        Transfer a full SE(3) grasp pose after mesh deformation.

        Args:
            T_grasp_old: 4x4 SE(3), or dict/format accepted by grasp_guess_to_SE3.
                Hand TCP pose in original mesh frame.

            anchor:
                Output from closest_surface_anchor(...).
                Usually built from old TCP translation:
                    anchor = make_grasp_anchor_from_SE3(T_grasp_old)

        Returns:
            T_grasp_new: 4x4 SE(3)
                Hand TCP pose in deformed mesh frame.

            debug_info: dict

        Formula:
            R_new = R_patch @ R_old
            t_new = anchor_new + R_patch @ (t_old - anchor_old)
        """
        if self.V_opt is None:
            raise ValueError("No optimized vertices found. Run slippage_reshape() first.")

        T_grasp_old = self.grasp_guess_to_SE3(T_grasp_old, quat_order=quat_order)
        R_old = T_grasp_old[:3, :3]
        t_old = T_grasp_old[:3, 3]

        face_id = int(anchor["face_id"])
        bary = np.asarray(anchor["barycentric"], dtype=np.float64)
        patch_ids = np.asarray(anchor["patch_vertex_ids"], dtype=np.int64)

        if len(patch_ids) < 3:
            raise ValueError("Anchor patch must contain at least 3 vertices")

        P = self.V[patch_ids]
        Q = self.V_opt[patch_ids]

        anchor_point_old = self._barycentric_point(self.V, self.F, face_id, bary)
        anchor_point_new = self._barycentric_point(self.V_opt, self.F, face_id, bary)

        if use_distance_weights:
            weights = self._patch_weights_from_anchor(
                patch_ids=patch_ids,
                anchor_point_old=anchor_point_old,
            )
        else:
            weights = None

        # Local shape matching:
        #   Q ~= R_patch @ P + t_patch
        R_patch, t_patch = self.fit_rigid_transform(P, Q, weights=weights)
        R_patch = self._project_to_SO3(R_patch)

        T_grasp_new = np.eye(4, dtype=np.float64)
        T_grasp_new[:3, :3] = self._project_to_SO3(R_patch @ R_old)
        T_grasp_new[:3, 3] = anchor_point_new + R_patch @ (t_old - anchor_point_old)

        Q_pred = P @ R_patch.T + t_patch
        per_vertex_error = np.linalg.norm(Q_pred - Q, axis=1)

        debug_info = {
            "R_patch": R_patch,
            "t_patch": t_patch,
            "anchor_point_old": anchor_point_old,
            "anchor_point_new": anchor_point_new,
            "fit_error_mean": float(per_vertex_error.mean()),
            "fit_error_max": float(per_vertex_error.max()),
            "num_patch_vertices": int(len(patch_ids)),
            "det_R_patch": float(np.linalg.det(R_patch)),
        }

        return T_grasp_new, debug_info

    def transfer_grasp_SE3(
        self,
        T_grasp_old=None,
        anchor=None,
        k_ring=2,
        use_distance_weights=True,
        quat_order="xyzw",
    ):
        """
        Public API for SE(3) grasp transfer.

        Args:
            T_grasp_old:
                If None, uses self.initial_grasp_guess.
                Otherwise can be 4x4 ndarray or supported dict.

            anchor:
                Optional precomputed anchor. If None, this function creates one
                from the old TCP position.

        Returns:
            T_grasp_new, anchor, debug_info
        """
        T_grasp_old = self.grasp_guess_to_SE3(T_grasp_old, quat_order=quat_order)

        if anchor is None:
            anchor = self.closest_surface_anchor(
                point=T_grasp_old[:3, 3],
                k_ring=k_ring,
            )

        T_grasp_new, debug_info = self.transfer_grasp_SE3_by_anchor(
            T_grasp_old=T_grasp_old,
            anchor=anchor,
            use_distance_weights=use_distance_weights,
            quat_order=quat_order,
        )

        return T_grasp_new, anchor, debug_info

    def transfer_initial_grasp_guess(
        self,
        anchor=None,
        k_ring=2,
        use_distance_weights=True,
        quat_order="xyzw",
        return_format="dict",
    ):
        """
        Convenience wrapper for self.initial_grasp_guess.

        return_format:
            "T":    return 4x4 SE(3)
            "dict": return dict with T_mesh_hand_tcp, t, R, quat
        """
        T_new, anchor, debug_info = self.transfer_grasp_SE3(
            T_grasp_old=None,
            anchor=anchor,
            k_ring=k_ring,
            use_distance_weights=use_distance_weights,
            quat_order=quat_order,
        )

        if return_format == "T":
            return T_new, anchor, debug_info

        if return_format != "dict":
            raise ValueError("return_format must be 'T' or 'dict'")

        result = {
            "T_mesh_hand_tcp": T_new,
            "t": T_new[:3, 3].copy(),
            "R": T_new[:3, :3].copy(),
            "quat": self._R_to_quat(T_new[:3, :3], order=quat_order),
        }

        return result, anchor, debug_info

    def write_transferred_grasp(self, output_path, grasp_result):
        """
        Save transferred grasp result to .npz.

        grasp_result can be:
            - 4x4 T
            - dict returned by transfer_initial_grasp_guess(return_format="dict")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(grasp_result, dict):
            np.savez(
                output_path,
                T_mesh_hand_tcp=np.asarray(grasp_result["T_mesh_hand_tcp"], dtype=np.float64),
                t=np.asarray(grasp_result["t"], dtype=np.float64),
                R=np.asarray(grasp_result["R"], dtype=np.float64),
                quat=np.asarray(grasp_result["quat"], dtype=np.float64),
            )
        else:
            T = np.asarray(grasp_result, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"grasp_result must be dict or 4x4 T, got {T.shape}")
            np.save(output_path, T)

    # ============================================================
    # Visualization
    # ============================================================

    def _make_o3d_mesh_from_vertices(self, vertices, color=(0.75, 0.75, 0.75)):
        """
        Build an Open3D TriangleMesh from given vertices and self.F.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.shape != self.V.shape:
            raise ValueError(
                f"vertices must have shape {self.V.shape}, got {vertices.shape}"
            )

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.F.astype(np.int32))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color)
        return mesh_o3d

    @staticmethod
    def _make_pose_axis(T, axis_size=0.05):
        """
        Create an RGB coordinate frame at SE(3) pose T.

        Open3D convention:
            x-axis: red
            y-axis: green
            z-axis: blue
        """
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T must be 4x4, got {T.shape}")

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(axis_size),
            origin=np.zeros(3),
        )
        axis.transform(T)
        return axis

    @staticmethod
    def _make_sphere(center, radius, color=(1.0, 0.2, 0.2)):
        """
        Small marker sphere for anchor/contact visualization.
        """
        center = np.asarray(center, dtype=np.float64).reshape(3)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
        sphere.translate(center)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        return sphere

    def _default_vis_scale(self):
        """
        Compute a reasonable visualization scale from the deformed mesh bbox.
        """
        if self.V_opt is not None:
            V_show = np.asarray(self.V_opt, dtype=np.float64)
        else:
            V_show = np.asarray(self.V, dtype=np.float64)

        bbox_min = V_show.min(axis=0)
        bbox_max = V_show.max(axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        if diag < 1e-12:
            diag = 1.0
        return diag

    def visualize_deformed_grasp_pose(
        self,
        T_grasp_new=None,
        anchor=None,
        debug_info=None,
        k_ring=2,
        use_distance_weights=True,
        quat_order="xyzw",
        axis_size=None,
        show_anchor=True,
        show_patch=True,
        show_old_grasp=False,
        T_grasp_old=None,
        window_name="Deformed mesh with transferred grasp pose",
        return_geometries=False,
    ):
        """
        Visualize deformed mesh + transferred SE(3) grasp pose.

        Main use:
            T_new, anchor, debug = augmentor.transfer_initial_grasp_guess(
                k_ring=2,
                return_format="T",
            )

            augmentor.visualize_deformed_grasp_pose(
                T_grasp_new=T_new,
                anchor=anchor,
                debug_info=debug,
            )

        If T_grasp_new is None, this function will automatically call
        transfer_grasp_SE3(...) using self.initial_grasp_guess.

        Visual elements:
            - deformed mesh: self.V_opt
            - RGB axis frame: transferred grasp TCP pose
            - optional red sphere: deformed surface anchor
            - optional patch vertices: small blue-ish spheres on deformed mesh
            - optional old grasp axis: original grasp pose, for comparison
        """
        if self.V_opt is None:
            raise ValueError(
                "No optimized vertices found. Run slippage_reshape() or "
                "displacement_reshape() before visualization."
            )

        # Auto-compute transferred grasp pose if not provided.
        if T_grasp_new is None:
            T_grasp_new, anchor, debug_info = self.transfer_grasp_SE3(
                T_grasp_old=None,
                anchor=anchor,
                k_ring=k_ring,
                use_distance_weights=use_distance_weights,
                quat_order=quat_order,
            )
        else:
            T_grasp_new = self.grasp_guess_to_SE3(
                T_grasp_new,
                quat_order=quat_order,
            )

        scale = self._default_vis_scale()
        if axis_size is None:
            axis_size = 0.08 * scale
        marker_radius = 0.012 * scale

        geoms = []

        # 1. Deformed mesh.
        mesh_def = self._make_o3d_mesh_from_vertices(
            self.V_opt,
            color=(0.72, 0.72, 0.72),
        )
        geoms.append(mesh_def)

        # 2. New grasp pose axis.
        grasp_axis = self._make_pose_axis(T_grasp_new, axis_size=axis_size)
        geoms.append(grasp_axis)

        # 3. Optional anchor marker and patch vertices.
        if anchor is not None and show_anchor:
            face_id = int(anchor["face_id"])
            bary = np.asarray(anchor["barycentric"], dtype=np.float64)
            anchor_point_new = self._barycentric_point(self.V_opt, self.F, face_id, bary)
            anchor_sphere = self._make_sphere(
                anchor_point_new,
                radius=marker_radius,
                color=(1.0, 0.1, 0.1),
            )
            geoms.append(anchor_sphere)

        if anchor is not None and show_patch:
            patch_ids = np.asarray(anchor["patch_vertex_ids"], dtype=np.int64)
            patch_points = self.V_opt[patch_ids]

            # Use a point cloud for patch vertices to keep it lightweight.
            patch_pcd = o3d.geometry.PointCloud()
            patch_pcd.points = o3d.utility.Vector3dVector(patch_points)
            patch_pcd.paint_uniform_color((0.1, 0.3, 1.0))
            geoms.append(patch_pcd)

        # 4. Optional old grasp pose axis, shown in original mesh frame.
        # This is only for visual comparison. It is not transformed to V_opt.
        if show_old_grasp:
            if T_grasp_old is None:
                T_grasp_old = self.grasp_guess_to_SE3(
                    None,
                    quat_order=quat_order,
                )
            else:
                T_grasp_old = self.grasp_guess_to_SE3(
                    T_grasp_old,
                    quat_order=quat_order,
                )
            old_axis = self._make_pose_axis(T_grasp_old, axis_size=0.75 * axis_size)
            geoms.append(old_axis)

        if debug_info is not None:
            print("Grasp transfer visualization debug:")
            if "fit_error_mean" in debug_info:
                print(f"  fit_error_mean: {debug_info['fit_error_mean']:.6g}")
            if "fit_error_max" in debug_info:
                print(f"  fit_error_max:  {debug_info['fit_error_max']:.6g}")
            if "num_patch_vertices" in debug_info:
                print(f"  num_patch_vertices: {debug_info['num_patch_vertices']}")

        if return_geometries:
            return geoms

        o3d.visualization.draw_geometries(
            geoms,
            window_name=window_name,
            width=1280,
            height=900,
        )
        return None
