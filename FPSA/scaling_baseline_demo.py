from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import yaml

from FPSA_baseline import ShapeAugmentor


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def dump_yaml_or_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_builtin(data)

    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _first_present(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _matrix4(data: Dict[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    value = _first_present(data, keys)
    if value is None:
        return None

    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{keys[0]} must be a 4x4 matrix, got {mat.shape}")
    return mat


def write_grasp_yaml(path: str | Path, grasp_result: Dict[str, Any], mesh_path: str | Path) -> None:
    """Write the transferred task/grasp frame in the existing loader format."""
    if not isinstance(grasp_result, dict):
        raise TypeError(f"grasp_result must be a dict, got {type(grasp_result).__name__}")

    T_mesh_hand_tcp = _matrix4(grasp_result, ["T_mesh_hand_tcp"])
    T_mesh_hand = _matrix4(grasp_result, ["T_mesh_hand"])

    # Existing loader expects both keys.  FPSA transfer may only return TCP, so
    # mirror the available transform rather than introducing new frame logic.
    if T_mesh_hand_tcp is None and T_mesh_hand is None:
        raise KeyError("grasp_result is missing T_mesh_hand_tcp / T_mesh_hand")
    if T_mesh_hand_tcp is None:
        T_mesh_hand_tcp = T_mesh_hand
    if T_mesh_hand is None:
        T_mesh_hand = T_mesh_hand_tcp

    opening = float(_first_present(grasp_result, ["opening_width_m", "opening"], 0.06))

    record = {
        "mesh_path": str(mesh_path),
        "reference_frame": str(_first_present(grasp_result, ["reference_frame"], "mesh_local_frame")),
        "hand_frame": str(_first_present(grasp_result, ["hand_frame"], "panda_hand")),
        "opening_width_m": opening,
        "pregrasp_opening_width_m": float(
            _first_present(grasp_result, ["pregrasp_opening_width_m", "pregrasp_opening"], opening)
        ),
        "finger_joint_m": float(
            _first_present(grasp_result, ["finger_joint_m", "finger_joint"], opening / 2.0)
        ),
        "T_mesh_hand": T_mesh_hand,
        "T_mesh_hand_tcp": T_mesh_hand_tcp,
    }
    dump_yaml_or_json(path, record)


def to_builtin(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    return x


def _carry_gripper_metadata(augmentor: ShapeAugmentor, grasp_result: Dict[str, Any]) -> None:
    """Preserve non-pose grasp metadata without changing FPSA frame transfer."""
    source = getattr(augmentor, "initial_grasp_guess", None)
    if not isinstance(source, dict):
        return

    for key in (
        "reference_frame",
        "hand_frame",
        "opening_width_m",
        "pregrasp_opening_width_m",
        "finger_joint_m",
        "opening",
        "pregrasp_opening",
        "finger_joint",
    ):
        if key in source and key not in grasp_result:
            grasp_result[key] = source[key]


# -----------------------------------------------------------------------------
# Baseline demo
# -----------------------------------------------------------------------------

def run_baseline(
    obj_path: str | Path,
    initial_grasp_path: str | Path,
    scale: float,
    output_dir: str | Path,
    patch_method: str = "k_ring",
    k_ring: int = 2,
    visualize: bool = False,
) -> Dict[str, Any]:
    obj_path = Path(obj_path)
    initial_grasp_path = Path(initial_grasp_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentor = ShapeAugmentor(
        obj_path=obj_path,
        initial_grasp_path=initial_grasp_path,
    )

    # 1) Baseline deformation: the only baseline-specific operation.
    augmentor.naive_uniform_scale(scale)

    scale_tag = f"{scale:.6g}".replace(".", "p")
    mesh_output = output_dir / f"{obj_path.stem}_uniform_scale_{scale_tag}.obj"

    # 2) Reuse the original FPSA collision-proxy transfer unchanged.
    mesh_path, coacd_path = augmentor.write_augment_obj(
        mesh_output,
        write_coacd=True,
        return_paths=True,
    )

    # 3) Reuse the original FPSA task/grasp-frame transfer unchanged.
    grasp_result, anchor, transfer_debug = augmentor.transfer_initial_grasp_guess(
        k_ring=k_ring,
        patch_method=patch_method,
        return_format="dict",
    )
    _carry_gripper_metadata(augmentor, grasp_result)

    grasp_output = output_dir / f"{obj_path.stem}_uniform_scale_{scale_tag}_grasp.yaml"
    write_grasp_yaml(grasp_output, grasp_result, mesh_path=mesh_path)

    debug_output = output_dir / f"{obj_path.stem}_uniform_scale_{scale_tag}_debug.json"
    debug_record = {
        "baseline": "naive_uniform_scaling",
        "scale_xyz": [float(scale), float(scale), float(scale)],
        "scale_center": "mesh_local_origin",
        "input_mesh": str(obj_path),
        "input_grasp": str(initial_grasp_path),
        "output_mesh": str(mesh_path),
        "output_collision_proxy": str(coacd_path),
        "output_grasp": str(grasp_output),
        "anchor": anchor,
        "transfer_debug": transfer_debug,
    }
    dump_yaml_or_json(debug_output, debug_record)

    if visualize:
        augmentor.visualize_deformed_grasp_pose(
            T_grasp_new=grasp_result["T_mesh_hand_tcp"],
            anchor=anchor,
            debug_info=transfer_debug,
            show_old_grasp=False,
        )

    return debug_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Naive uniform-scaling baseline using the original FPSA collision "
            "proxy and task-frame transfer logic."
        )
    )
    parser.add_argument("--obj", required=True, help="Input object OBJ mesh.")
    parser.add_argument("--grasp", required=True, help="Initial grasp/task-frame file consumed by FPSA.")
    parser.add_argument("--scale", type=float, default=1.2, help="Uniform x/y/z scale factor (default: 1.2).")
    parser.add_argument("--output-dir", default="outputs/uniform_scaling_baseline")
    parser.add_argument(
        "--patch-method",
        choices=["k_ring", "xyz"],
        default="k_ring",
        help="Existing FPSA task-frame transfer patch method.",
    )
    parser.add_argument("--k-ring", type=int, default=2)
    parser.add_argument("--visualize", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    result = run_baseline(
        obj_path=args.obj,
        initial_grasp_path=args.grasp,
        scale=args.scale,
        output_dir=args.output_dir,
        patch_method=args.patch_method,
        k_ring=args.k_ring,
        visualize=args.visualize,
    )

    print("[baseline] naive uniform scaling finished")
    print(f"  mesh:            {result['output_mesh']}")
    print(f"  collision proxy: {result['output_collision_proxy']}")
    print(f"  transferred pose:{result['output_grasp']}")
