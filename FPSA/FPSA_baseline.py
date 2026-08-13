"""Naive uniform-scaling baseline built on top of the original FPSA pipeline.

This module deliberately keeps all collision-proxy and task/grasp-frame transfer
logic in the original :mod:`FPSA` implementation.  The only new behavior is
naive mesh-axis scaling that populates ``V_opt`` / ``V_work`` exactly like an
FPSA reshape result.  ``naive_uniform_scale(scale)`` keeps the original isotropic
xyz baseline, while ``naive_axis_uniform_scale(scale, axes=...)`` supports the
x/y/z/xy/xz/yz/xyz batch-baseline groups using one shared scale on selected axes.
"""

from __future__ import annotations

import numpy as np

from FPSA import ShapeAugmentor as _FPSAShapeAugmentor


class ShapeAugmentor(_FPSAShapeAugmentor):
    """FPSA-compatible augmentor with a naive uniform-scaling baseline."""

    @staticmethod
    def _axis_scale_vector(scale: float, axes=("x", "y", "z")) -> np.ndarray:
        """Build [sx, sy, sz] using one shared scalar on the selected axes."""
        scale = float(scale)
        if not np.isfinite(scale):
            raise ValueError(f"scale must be finite, got {scale}")
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale}")

        if isinstance(axes, str):
            key = axes.lower().strip().replace("+", "")
            axes = list(key)
        else:
            axes = [str(a).lower().strip() for a in axes]

        valid = {"x", "y", "z"}
        unknown = sorted(set(axes) - valid)
        if unknown:
            raise ValueError(f"Unknown scaling axes {unknown}; supported axes are x/y/z")
        if len(axes) == 0:
            raise ValueError("At least one scaling axis is required")

        factors = np.ones(3, dtype=np.float64)
        for axis in axes:
            factors[{"x": 0, "y": 1, "z": 2}[axis]] = scale
        return factors

    def naive_axis_uniform_scale(self, scale: float, axes=("x", "y", "z")):
        """Naively scale selected mesh axes using one shared scale factor.

        Examples::

            axes="x"   -> [scale, 1, 1]
            axes="xy"  -> [scale, scale, 1]
            axes="xyz" -> [scale, scale, scale]

        Scaling is about the mesh-local origin and only populates the same
        ``V_opt`` / ``V_work`` state used by FPSA reshape methods.  Collision
        proxy transfer and task/grasp-frame transfer remain inherited from the
        original FPSA implementation.
        """
        scale_xyz = self._axis_scale_vector(scale, axes=axes)
        V_in = np.asarray(self.V_work, dtype=np.float64)
        V_new = V_in * scale_xyz[None, :]

        self.V_opt = V_new.copy()
        self.V_work = V_new.copy()
        self.face_k1 = None
        self.face_k2 = None
        self._vertex_kdtree = None
        return self.V_opt

    def naive_uniform_scale(self, scale: float):
        """Uniformly scale x/y/z together by one scalar about mesh-local origin."""
        return self.naive_axis_uniform_scale(scale=scale, axes="xyz")

    # Explicit aliases using common baseline naming.
    naive_uniform_scaling = naive_uniform_scale
    uniform_scaling_baseline = naive_uniform_scale
