from __future__ import annotations

from icecream import ic

from FPSA import ShapeAugmentor
from gear_hole_constraint import GearHoleHardConstraint
import numpy as np
# ----------------------------
# IO helpers
# ----------------------------

import argparse
import yaml
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import json


def load_meta(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML meta files. Install with: pip install pyyaml")
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported meta file suffix: {path.suffix}. Use .yaml/.yml/.json")


def dump_yaml_or_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_builtin(data)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            path = path.with_suffix(".json")
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
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
    """Write the transferred grasp in the format consumed by load_initial_grasp_pose()."""
    if not isinstance(grasp_result, dict):
        raise TypeError(f"grasp_result must be a dict, got {type(grasp_result).__name__}")

    T_mesh_hand_tcp = _matrix4(grasp_result, ["T_mesh_hand_tcp"])
    T_mesh_hand = _matrix4(grasp_result, ["T_mesh_hand"])

    # The user's loader expects both keys to exist, but the FPSA transfer result may
    # only return T_mesh_hand_tcp. Do not compute an extra transform here; just mirror
    # the available 4x4 matrix so the YAML remains loadable by the existing reader.
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
        "finger_joint_m": float(_first_present(grasp_result, ["finger_joint_m", "finger_joint"], opening / 2.0)),
        "T_mesh_hand": T_mesh_hand,
        "T_mesh_hand_tcp": T_mesh_hand_tcp,
    }
    dump_yaml_or_json(path, record)


def to_builtin(x: Any) -> Any:
    """Convert numpy objects to JSON/YAML-safe Python objects."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    return x

if __name__ == "__main__":
    obj_path = "/home/iadc/GeoBridge/data/objects/gear_extraction/gear/gear.obj"
    initial_grasp_path = "/home/iadc/GeoBridge/data/objects/gear_extraction/gear/grasp.yaml"

    augmentor = ShapeAugmentor(
        obj_path=obj_path,
        initial_grasp_path=initial_grasp_path,
    )

    hole_constraint = GearHoleHardConstraint.from_augmentor(
        augmentor,
        axis="z",  # 如果齿轮位于 XY 平面、厚度方向是 Z
        center=[0.0, 0.0, 0.0],
        radius=0.0075,
        target_mode="original",
        component_mode="all",
        radial_tolerance=0.002,
    )
    print(hole_constraint.topology_report(augmentor.F))
    print(hole_constraint.target_displacement_report(augmentor.V))
    # ============================================================
    # Step 2: slippage reshaping
    # ============================================================
    # This starts from the original mesh. After this call, augmentor.V_work is
    # automatically updated to the slippage result, so the next APAP call will
    # use the slippage-deformed mesh as input.
    constraint_ids = [260, 257, 5437, 255, 469, 468, 467, 466, 464, 462, 459, 455,
      453, 450, 447, 275, 272, 269, 265, 263, 267, 1, 451, 452, 460, 461, 465, 258,
      259, 254, 270, 271, 273, 274, 276, 644, 648, 5463, 5405, 5454, 5416, 5320, 5170,
      5359, 5184, 5219, 5261, 5289, 5334]
    apap_move_ids = [5320, 5170, 5359, 5184, 5219, 5261, 5289, 5334]
    apap_refine_displacements = np.array([
      [0.0, -1.0, 0.0],
      [0.0, 1.0, 0.0],
      [-1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.7071067811865476, 0.7071067811865476, 0.0],
      [0.7071067811865476, -0.7071067811865476, 0.0],
      [-0.7071067811865476, -0.7071067811865476, 0.0],
      [-0.7071067811865476, 0.7071067811865476, 0.0],
    ])

    apap_refine_displacements = 0.06 * apap_refine_displacements

    V_final = hole_constraint.displacement_reshape(
        augmentor=augmentor,
        constraint_ids=constraint_ids,
        displace_idxs=apap_move_ids,
        displacements=apap_refine_displacements,
        max_iters=120,
        reshape_method="slippage",
        input_name="step02_slippage_hard_hole",
        post_enforce=True,

    )
    print(hole_constraint.diameter_report(V_final))

    augmentor.write_augment_obj(
        output_path="step02_slippage.obj",
        write_coacd=True,
    )

    print("V_final:", V_final.shape)

    # ============================================================
    # Grasp transfer and visualization
    # ============================================================
    # self.V is still the original reference mesh. self.V_opt is the final mesh
    # after slippage -> APAP, so grasp transfer maps the original grasp to the
    # final chained deformation.


    T_new, anchor, debug = augmentor.transfer_initial_grasp_guess(
        k_ring=3,
        use_distance_weights=True,
        quat_order="xyzw",
        patch_method="k_ring",
    )
    ic(T_new)
    augmentor.visualize_deformed_grasp_pose(
        T_grasp_new=T_new["T_mesh_hand_tcp"],
        anchor=anchor,
        debug_info=debug,
        show_anchor=True,
        show_patch=True,
        show_old_grasp=True,
    )
    

    # write_grasp_yaml("transferred_grasp.yaml",T_new, mesh_path="step02_slippage.obj" )

    

