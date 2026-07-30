"""Hard constraint for keeping a gear's center hole unchanged during FPSA/APAP deformation.

This module does not modify FPSA.py.  It wraps an existing ShapeAugmentor and:

1. automatically finds the vertices on the cylindrical inner wall;
2. fits an exact reference cylinder to those vertices;
3. constrains every geometric copy of the inner-hole shell, including OBJ seam
   duplicates that are coincident in space but disconnected in topology;
4. preserves the original hole coordinates by default, then writes those exact
   coordinates back after the solver finishes.  An optional ``target_mode="cylinder"``
   is available when the source hole itself should first be circularized.

Typical use
-----------

    from FPSA import ShapeAugmentor
    from gear_hole_constraint import GearHoleHardConstraint

    augmentor = ShapeAugmentor(obj_path=obj_path, initial_grasp_path=grasp_path)

    hole = GearHoleHardConstraint.from_augmentor(
        augmentor,
        axis="auto",       # for a flat spur gear, auto chooses the thickness axis
        # radius=0.010,     # optional: provide this if automatic detection is ambiguous
        # center=[0, 0, 0], # optional: any point on the desired cylinder axis
    )

    V_final = hole.displacement_reshape(
        augmentor=augmentor,
        constraint_ids=constraint_ids,
        displace_idxs=move_ids,
        displacements=displacements,
        max_iters=120,
        reshape_method="slippage",
        input_name="gear_hard_hole",
    )

    print(hole.diameter_report(V_final))

The default mode pins the complete inner cylindrical wall, not only its radius.
That is intentional: it is the strongest positional hard constraint supported by
FPSA/APAP's current API and prevents center drift, diameter drift, ovalization,
twist, and axial distortion of the hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import argparse
import json

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]
AxisLike = Union[str, int, ArrayLike]


@dataclass
class HoleConstraintSpec:
    """Geometry and vertex correspondence of the fixed reference cylinder."""

    vertex_ids: np.ndarray
    center: np.ndarray
    axis: np.ndarray
    radius: float
    reference_targets: np.ndarray
    radial_tolerance: float
    target_mode: str


class GearHoleHardConstraint:
    """Keep the center hole fixed while the rest of a gear is deformed."""

    def __init__(self, spec: HoleConstraintSpec):
        self.spec = spec

    @property
    def vertex_ids(self) -> np.ndarray:
        return self.spec.vertex_ids

    @property
    def center(self) -> np.ndarray:
        return self.spec.center

    @property
    def axis(self) -> np.ndarray:
        return self.spec.axis

    @property
    def radius(self) -> float:
        return self.spec.radius

    @property
    def diameter(self) -> float:
        return 2.0 * self.spec.radius

    @classmethod
    def from_augmentor(
        cls,
        augmentor: Any,
        *,
        axis: AxisLike = "auto",
        center: Optional[ArrayLike] = None,
        radius: Optional[float] = None,
        radial_tolerance: Optional[float] = None,
        min_vertices: int = 8,
        target_mode: str = "original",
        component_mode: str = "all",
    ) -> "GearHoleHardConstraint":
        """Detect and fit the hole using augmentor.V and augmentor.F."""
        return cls.from_mesh(
            vertices=np.asarray(augmentor.V, dtype=np.float64),
            faces=np.asarray(augmentor.F, dtype=np.int64),
            axis=axis,
            center=center,
            radius=radius,
            radial_tolerance=radial_tolerance,
            min_vertices=min_vertices,
            target_mode=target_mode,
            component_mode=component_mode,
        )

    @classmethod
    def from_mesh(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        axis: AxisLike = "auto",
        center: Optional[ArrayLike] = None,
        radius: Optional[float] = None,
        radial_tolerance: Optional[float] = None,
        min_vertices: int = 8,
        target_mode: str = "original",
        component_mode: str = "all",
    ) -> "GearHoleHardConstraint":
        """Detect the innermost cylindrical vertex shell and fit a circle to it.

        Parameters
        ----------
        vertices, faces:
            Original gear mesh arrays.
        axis:
            ``"auto"``, ``"x"``, ``"y"``, ``"z"``, axis index 0/1/2, or a
            3-vector. ``"auto"`` chooses the smallest bounding-box dimension,
            which is normally the thickness direction of a spur gear.
        center:
            Optional point on the hole axis. If omitted, the bounding-box center
            is used for initial detection and then refined by circle fitting.
        radius:
            Optional expected hole radius. Supplying it is the most reliable way
            to disambiguate meshes that contain several coaxial cylindrical steps.
        radial_tolerance:
            Selection band around the hole radius. The automatic default is
            0.2 percent of the mesh bounding-box diagonal.
        min_vertices:
            Minimum accepted number of hole vertices.
        target_mode:
            ``"original"`` keeps the source hole exactly unchanged. This is the
            recommended mode and avoids creating a visible ring by circularizing
            only part of an unwelded OBJ seam. ``"cylinder"`` projects the selected
            vertices onto one fitted mathematical cylinder.
        component_mode:
            ``"all"`` keeps all candidate shell components. This is important for
            OBJ meshes whose wall/top/bottom surfaces use coincident duplicate
            vertices. ``"largest"`` reproduces the old topology-only behavior.
        """
        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)
        cls._validate_mesh_arrays(V, F)

        if min_vertices < 3:
            raise ValueError("min_vertices must be at least 3")

        target_mode = str(target_mode).lower()
        if target_mode not in {"original", "cylinder"}:
            raise ValueError("target_mode must be 'original' or 'cylinder'")

        component_mode = str(component_mode).lower()
        if component_mode not in {"all", "largest"}:
            raise ValueError("component_mode must be 'all' or 'largest'")

        axis_unit = cls._resolve_axis(V, axis)
        bbox_min = V.min(axis=0)
        bbox_max = V.max(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
        if bbox_diag <= 0.0:
            raise ValueError("Mesh bounding box is degenerate")

        center0 = (
            0.5 * (bbox_min + bbox_max)
            if center is None
            else np.asarray(center, dtype=np.float64).reshape(3)
        )

        tol = (
            max(1e-10, 2.0e-3 * bbox_diag)
            if radial_tolerance is None
            else float(radial_tolerance)
        )
        if tol <= 0.0:
            raise ValueError("radial_tolerance must be positive")

        radii0 = cls._radial_distances(V, center0, axis_unit)
        if radius is None:
            radius0 = cls._estimate_innermost_shell_radius(
                radii0,
                tolerance=tol,
                min_vertices=min_vertices,
            )
        else:
            radius0 = float(radius)
            if radius0 <= 0.0:
                raise ValueError("radius must be positive")

        candidate = np.flatnonzero(np.abs(radii0 - radius0) <= tol)
        if component_mode == "largest":
            candidate = cls._largest_connected_component(candidate, F)
        if candidate.size < min_vertices:
            raise ValueError(
                "Could not find enough center-hole vertices. "
                f"Found {candidate.size}, expected at least {min_vertices}. "
                "Provide axis, center, radius, or a larger radial_tolerance explicitly."
            )

        # First fit refines an imperfect bbox-center estimate.
        fitted_center, fitted_radius = cls._fit_circle_in_axis_plane(
            V[candidate],
            axis=axis_unit,
            center_hint=center0,
        )

        # Re-select around the fitted circle. This removes nearby vertices from a
        # top/bottom face while retaining all rings of the cylindrical side wall.
        radii1 = cls._radial_distances(V, fitted_center, axis_unit)
        fitted_tol = max(tol, 5.0e-4 * fitted_radius)
        candidate2 = np.flatnonzero(np.abs(radii1 - fitted_radius) <= fitted_tol)
        if component_mode == "largest":
            candidate2 = cls._largest_connected_component(candidate2, F)
        if candidate2.size >= min_vertices:
            candidate = candidate2
            fitted_center, fitted_radius = cls._fit_circle_in_axis_plane(
                V[candidate],
                axis=axis_unit,
                center_hint=fitted_center,
            )

        if target_mode == "original":
            # Hard-pin the actual source hole coordinates.  This preserves the
            # original polygonization, chamfer, axial rings, and duplicated seam
            # vertices exactly, so post-enforcement cannot create a new protruding
            # circular ring.
            reference_targets = V[candidate].copy()
        else:
            reference_targets = cls._project_points_to_cylinder(
                points=V[candidate],
                center=fitted_center,
                axis=axis_unit,
                radius=fitted_radius,
                fallback_points=V[candidate],
            )

        spec = HoleConstraintSpec(
            vertex_ids=np.asarray(candidate, dtype=np.int64),
            center=np.asarray(fitted_center, dtype=np.float64),
            axis=np.asarray(axis_unit, dtype=np.float64),
            radius=float(fitted_radius),
            reference_targets=np.asarray(reference_targets, dtype=np.float64),
            radial_tolerance=float(fitted_tol),
            target_mode=target_mode,
        )
        return cls(spec)

    def displacement_reshape(
        self,
        augmentor: Any,
        constraint_ids: Iterable[int],
        displace_idxs: Iterable[int],
        displacements: np.ndarray,
        *,
        max_iters: int = 20,
        handle_error_distrib_enabled: bool = False,
        input_name: Optional[str] = None,
        reshape_method: str = "slippage",
        post_enforce: bool = True,
    ) -> np.ndarray:
        """Run FPSA/APAP with the complete hole added as a hard constraint.

        This mirrors ``ShapeAugmentor.displacement_reshape`` but builds explicit
        target positions. In the default ``target_mode="original"``, those targets
        are the exact source-hole coordinates rather than a newly fitted circle.
        The supplied ``displace_idxs`` must not contain a hole vertex.
        """
        V_in = np.asarray(augmentor.V_work, dtype=np.float64)
        if V_in.ndim != 2 or V_in.shape[1] != 3:
            raise ValueError("augmentor.V_work must have shape (N, 3)")
        if int(self.vertex_ids.max()) >= len(V_in):
            raise ValueError("Hole vertex ids do not match this augmentor topology")

        user_constraint_ids = [int(v) for v in constraint_ids]
        move_ids = [int(v) for v in displace_idxs]
        disp = np.asarray(displacements, dtype=np.float64)
        if disp.ndim == 1:
            if len(move_ids) != 1 or disp.shape != (3,):
                raise ValueError(
                    "A 1D displacement requires exactly one displace_idx and shape (3,)"
                )
            disp = disp.reshape(1, 3)
        if disp.shape != (len(move_ids), 3):
            raise ValueError(
                f"displacements must have shape {(len(move_ids), 3)}, got {disp.shape}"
            )

        hole_set = set(map(int, self.vertex_ids))
        overlap = sorted(hole_set.intersection(move_ids))
        if overlap:
            raise ValueError(
                "A moved handle cannot also be a fixed hole vertex. "
                f"Overlapping ids: {overlap[:16]}"
            )

        # Dictionary insertion order gives deterministic solver input. User fixed
        # handles are inserted first, then hole targets overwrite any duplicates.
        target_by_id: Dict[int, np.ndarray] = {}
        for vid in user_constraint_ids:
            if vid < 0 or vid >= len(V_in):
                raise IndexError(f"constraint vertex id out of range: {vid}")
            target_by_id[vid] = V_in[vid].copy()

        for vid, delta in zip(move_ids, disp):
            if vid not in target_by_id:
                raise ValueError(
                    f"displace_idxs must be a subset of constraint_ids, got {vid}"
                )
            target_by_id[vid] = target_by_id[vid] + delta

        for row, vid in enumerate(self.vertex_ids):
            target_by_id[int(vid)] = self.spec.reference_targets[row].copy()

        merged_ids = list(target_by_id.keys())
        merged_targets = np.vstack([target_by_id[vid] for vid in merged_ids])

        method = str(reshape_method).lower()
        if method == "slippage":
            V_new = augmentor.slippage_reshape(
                constraint_ids=merged_ids,
                target_positions=merged_targets,
                max_iters=max_iters,
                handle_error_distrib_enabled=handle_error_distrib_enabled,
                input_name=input_name,
            )
        elif method in {"apap", "arap"}:
            V_new = augmentor.APAP_reshape(
                constraint_ids=merged_ids,
                target_positions=merged_targets,
                max_iters=max_iters,
                handle_error_distrib_enabled=handle_error_distrib_enabled,
                input_name=input_name,
            )
        else:
            raise ValueError(
                f"Unknown reshape_method={reshape_method!r}; use 'slippage' or 'apap'"
            )

        if post_enforce:
            V_new = self.enforce_on_augmentor(augmentor, vertices=V_new)
        return np.asarray(V_new, dtype=np.float64)

    def enforce(self, vertices: np.ndarray) -> np.ndarray:
        """Return a copy with hole vertices set exactly to their reference targets."""
        V = np.asarray(vertices, dtype=np.float64)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if int(self.vertex_ids.max()) >= len(V):
            raise ValueError("Hole vertex ids do not match the supplied topology")
        out = V.copy()
        out[self.vertex_ids] = self.spec.reference_targets
        return out

    def enforce_on_augmentor(
        self,
        augmentor: Any,
        *,
        vertices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Enforce the hole targets and synchronize ShapeAugmentor working state."""
        source = augmentor.V_opt if vertices is None else vertices
        if source is None:
            raise ValueError("No deformed vertices are available to enforce")
        V_exact = self.enforce(np.asarray(source, dtype=np.float64))

        augmentor.V_opt = V_exact.copy()
        augmentor.V_work = V_exact.copy()
        if hasattr(augmentor, "face_k1"):
            augmentor.face_k1 = None
        if hasattr(augmentor, "face_k2"):
            augmentor.face_k2 = None
        if hasattr(augmentor, "_vertex_kdtree"):
            augmentor._vertex_kdtree = None
        return V_exact

    def diameter_report(self, vertices: np.ndarray) -> Dict[str, float]:
        """Measure radius/diameter drift on the constrained hole vertices."""
        V = np.asarray(vertices, dtype=np.float64)
        r = self._radial_distances(V[self.vertex_ids], self.center, self.axis)
        r_ref = self._radial_distances(
            self.spec.reference_targets, self.center, self.axis
        )
        radial_error = np.abs(r - r_ref)
        return {
            "target_mode": self.spec.target_mode,
            "fitted_reference_radius": float(self.radius),
            "fitted_reference_diameter": float(self.diameter),
            "reference_radius_min": float(r_ref.min()),
            "reference_radius_mean": float(r_ref.mean()),
            "reference_radius_max": float(r_ref.max()),
            "measured_radius_min": float(r.min()),
            "measured_radius_mean": float(r.mean()),
            "measured_radius_max": float(r.max()),
            "max_radius_drift": float(radial_error.max()),
            "max_diameter_drift": float(2.0 * radial_error.max()),
            "num_hole_vertices": int(len(self.vertex_ids)),
        }

    def topology_report(self, faces: np.ndarray) -> Dict[str, Any]:
        """Report whether the selected hole shell contains disconnected OBJ seams."""
        F = np.asarray(faces, dtype=np.int64)
        ids = np.asarray(self.vertex_ids, dtype=np.int64)
        allowed = set(map(int, ids))
        adjacency: Dict[int, set[int]] = {int(i): set() for i in ids}
        for tri in F:
            a, b, c = map(int, tri)
            for i, j in ((a, b), (b, c), (c, a)):
                if i in allowed and j in allowed:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        sizes = []
        unseen = set(allowed)
        while unseen:
            seed = unseen.pop()
            size = 1
            stack = [seed]
            while stack:
                current = stack.pop()
                neighbors = adjacency[current].intersection(unseen)
                for nxt in neighbors:
                    unseen.remove(nxt)
                    stack.append(nxt)
                    size += 1
            sizes.append(size)
        sizes.sort(reverse=True)
        return {
            "target_mode": self.spec.target_mode,
            "num_hole_vertices": int(len(ids)),
            "num_topological_components": int(len(sizes)),
            "component_sizes": sizes,
        }

    def target_displacement_report(self, original_vertices: np.ndarray) -> Dict[str, float]:
        """Measure how far construction of the reference targets moves the source mesh."""
        V = np.asarray(original_vertices, dtype=np.float64)
        delta = np.linalg.norm(V[self.vertex_ids] - self.spec.reference_targets, axis=1)
        return {
            "target_mode": self.spec.target_mode,
            "mean_reference_target_shift": float(delta.mean()),
            "max_reference_target_shift": float(delta.max()),
        }

    def save_spec(self, path: Union[str, Path]) -> None:
        """Save detected ids and fitted cylinder parameters as JSON."""
        data = {
            "vertex_ids": self.vertex_ids.tolist(),
            "center": self.center.tolist(),
            "axis": self.axis.tolist(),
            "radius": float(self.radius),
            "diameter": float(self.diameter),
            "radial_tolerance": float(self.spec.radial_tolerance),
            "target_mode": self.spec.target_mode,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _validate_mesh_arrays(V: np.ndarray, F: np.ndarray) -> None:
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"vertices must have shape (N, 3), got {V.shape}")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {F.shape}")
        if len(V) == 0 or len(F) == 0:
            raise ValueError("Mesh must contain vertices and triangular faces")
        if F.min() < 0 or F.max() >= len(V):
            raise ValueError("faces contain invalid vertex indices")

    @staticmethod
    def _resolve_axis(V: np.ndarray, axis: AxisLike) -> np.ndarray:
        if isinstance(axis, str):
            key = axis.lower()
            if key == "auto":
                extents = V.max(axis=0) - V.min(axis=0)
                idx = int(np.argmin(extents))
                out = np.eye(3, dtype=np.float64)[idx]
            elif key in {"x", "y", "z"}:
                out = np.eye(3, dtype=np.float64)[{"x": 0, "y": 1, "z": 2}[key]]
            else:
                raise ValueError("axis string must be 'auto', 'x', 'y', or 'z'")
        elif isinstance(axis, (int, np.integer)):
            idx = int(axis)
            if idx not in {0, 1, 2}:
                raise ValueError("axis index must be 0, 1, or 2")
            out = np.eye(3, dtype=np.float64)[idx]
        else:
            out = np.asarray(axis, dtype=np.float64).reshape(3)

        norm = float(np.linalg.norm(out))
        if norm < 1e-12:
            raise ValueError("axis vector is too small")
        return out / norm

    @staticmethod
    def _axis_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        v /= np.linalg.norm(v)
        return u, v

    @staticmethod
    def _radial_distances(
        points: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
    ) -> np.ndarray:
        D = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
        axial = (D @ axis)[:, None] * axis[None, :]
        return np.linalg.norm(D - axial, axis=1)

    @staticmethod
    def _estimate_innermost_shell_radius(
        radii: np.ndarray,
        *,
        tolerance: float,
        min_vertices: int,
    ) -> float:
        r = np.asarray(radii, dtype=np.float64)
        r = np.sort(r[np.isfinite(r) & (r > 1e-12)])
        if len(r) < min_vertices:
            raise ValueError("Not enough nonzero radial samples")

        # Split sorted radii into shells using the same tolerance later used for
        # vertex selection. Return the first shell containing enough vertices.
        start = 0
        for i in range(1, len(r) + 1):
            at_end = i == len(r)
            has_gap = (not at_end) and ((r[i] - r[i - 1]) > tolerance)
            if at_end or has_gap:
                shell = r[start:i]
                if len(shell) >= min_vertices:
                    return float(np.median(shell))
                start = i

        # Fallback for noisy geometry with no clean radial gap.
        return float(np.median(r[:min_vertices]))

    @staticmethod
    def _largest_connected_component(vertex_ids: np.ndarray, F: np.ndarray) -> np.ndarray:
        ids = np.asarray(vertex_ids, dtype=np.int64)
        if ids.size == 0:
            return ids
        allowed = set(map(int, ids))
        adjacency: Dict[int, set[int]] = {int(i): set() for i in ids}
        for tri in F:
            a, b, c = map(int, tri)
            if a in allowed and b in allowed:
                adjacency[a].add(b)
                adjacency[b].add(a)
            if b in allowed and c in allowed:
                adjacency[b].add(c)
                adjacency[c].add(b)
            if c in allowed and a in allowed:
                adjacency[c].add(a)
                adjacency[a].add(c)

        best: list[int] = []
        unseen = set(allowed)
        while unseen:
            seed = unseen.pop()
            component = [seed]
            stack = [seed]
            while stack:
                current = stack.pop()
                neighbors = adjacency[current].intersection(unseen)
                for nxt in neighbors:
                    unseen.remove(nxt)
                    stack.append(nxt)
                    component.append(nxt)
            if len(component) > len(best):
                best = component
        return np.asarray(sorted(best), dtype=np.int64)

    @classmethod
    def _fit_circle_in_axis_plane(
        cls,
        points: np.ndarray,
        *,
        axis: np.ndarray,
        center_hint: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        P = np.asarray(points, dtype=np.float64)
        center_hint = np.asarray(center_hint, dtype=np.float64).reshape(3)
        u, v = cls._axis_basis(axis)
        D = P - center_hint
        x = D @ u
        y = D @ v

        # x^2 + y^2 = 2*cx*x + 2*cy*y + c0
        A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
        b = x * x + y * y
        solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 3:
            raise ValueError("Hole vertices are insufficient for stable circle fitting")
        cx, cy, c0 = solution
        radius_sq = float(c0 + cx * cx + cy * cy)
        if radius_sq <= 0.0:
            raise ValueError("Circle fitting produced a non-positive radius")

        fitted_center = center_hint + cx * u + cy * v
        fitted_radius = float(np.sqrt(radius_sq))
        return fitted_center, fitted_radius

    @staticmethod
    def _project_points_to_cylinder(
        points: np.ndarray,
        *,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        fallback_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        P = np.asarray(points, dtype=np.float64)
        C = np.asarray(center, dtype=np.float64).reshape(3)
        A = np.asarray(axis, dtype=np.float64).reshape(3)
        D = P - C
        height = D @ A
        radial = D - height[:, None] * A[None, :]
        radial_norm = np.linalg.norm(radial, axis=1)

        bad = radial_norm < 1e-12
        if np.any(bad):
            if fallback_points is None:
                raise ValueError("Cannot project a point lying on the cylinder axis")
            fallback = np.asarray(fallback_points, dtype=np.float64) - C
            fallback_height = fallback @ A
            fallback_radial = fallback - fallback_height[:, None] * A[None, :]
            radial[bad] = fallback_radial[bad]
            radial_norm = np.linalg.norm(radial, axis=1)
            if np.any(radial_norm < 1e-12):
                raise ValueError("Fallback points also lie on the cylinder axis")

        radial_unit = radial / radial_norm[:, None]
        return C + height[:, None] * A[None, :] + float(radius) * radial_unit


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect a gear center hole and save its hard-constraint specification."
    )
    parser.add_argument("obj", type=Path, help="Input gear OBJ")
    parser.add_argument("--axis", default="auto", help="auto, x, y, or z")
    parser.add_argument("--radius", type=float, default=None, help="Expected hole radius")
    parser.add_argument("--tolerance", type=float, default=None, help="Radial selection tolerance")
    parser.add_argument(
        "--target-mode", choices=["original", "cylinder"], default="original",
        help="Preserve source hole coordinates or circularize to a fitted cylinder",
    )
    parser.add_argument(
        "--component-mode", choices=["all", "largest"], default="all",
        help="Keep all unwelded OBJ shell components or only the largest one",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gear_hole_constraint.json"),
        help="Output JSON specification",
    )
    args = parser.parse_args()

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("CLI mode requires trimesh: pip install trimesh") from exc

    mesh = trimesh.load(args.obj, force="mesh", process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected one Trimesh, got {type(mesh).__name__}")

    hard_hole = GearHoleHardConstraint.from_mesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        axis=args.axis,
        radius=args.radius,
        radial_tolerance=args.tolerance,
        target_mode=args.target_mode,
        component_mode=args.component_mode,
    )
    hard_hole.save_spec(args.output)
    print(json.dumps({
        "output": str(args.output),
        "num_hole_vertices": int(len(hard_hole.vertex_ids)),
        "center": hard_hole.center.tolist(),
        "axis": hard_hole.axis.tolist(),
        "radius": hard_hole.radius,
        "diameter": hard_hole.diameter,
        "target_mode": hard_hole.spec.target_mode,
        "topology": hard_hole.topology_report(np.asarray(mesh.faces)),
        "target_shift": hard_hole.target_displacement_report(np.asarray(mesh.vertices)),
    }, indent=2))


if __name__ == "__main__":
    _main()