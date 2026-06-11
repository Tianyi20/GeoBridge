"""Python package wrapper for slippage-preserving reshaping."""

from .slippage_reshaping_cpp import (
    Options,
    Result,
    ReshapingOptimizer,
    optimize,
    optimize_from_edit_file,
    save_obj,
)


def optimize_mesh(
    vertices,
    faces,
    face_k1,
    face_k2,
    constraint_vertex_ids,
    constraint_positions,
    *,
    max_iters=100,
    handle_error_distrib_enabled=True,
    debug_folder="",
    input_name="python_binding",
):
    """Convenience API returning only optimized vertices.

    Parameters are numpy-compatible arrays:
        vertices: (#V, 3)
        faces: (#F, 3)
        face_k1: (#F,)
        face_k2: (#F,)
        constraint_vertex_ids: list[int]
        constraint_positions: (N, 3), absolute target positions
    """
    options = Options(
        max_iters=max_iters,
        handle_error_distrib_enabled=handle_error_distrib_enabled,
        debug_folder=debug_folder,
        input_name=input_name,
    )
    result = optimize(
        vertices,
        faces,
        face_k1,
        face_k2,
        list(constraint_vertex_ids),
        constraint_positions,
        options,
    )
    return result.vertices


__all__ = [
    "Options",
    "Result",
    "ReshapingOptimizer",
    "optimize",
    "optimize_mesh",
    "optimize_from_edit_file",
    "save_obj",
]
