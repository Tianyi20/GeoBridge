"""Batch generator for the naive axis-uniform scaling baseline.

This script mirrors the output contract of FPSA_batch_randomizer.py so the
resulting mesh/grasp pairs can be sampled by the existing PickUpSim FPSAObjectDR
path.  It intentionally does *not* modify the original FPSA deformation logic.

Baseline groups
---------------
A single scalar ``s`` is sampled with deterministic ``np.linspace`` over
``[scale_min, scale_max]`` independently for every selected axis group:

    x   -> [s, 1, 1]
    y   -> [1, s, 1]
    z   -> [1, 1, s]
    xy  -> [s, s, 1]
    xz  -> [s, 1, s]
    yz  -> [1, s, s]
    xyz -> [s, s, s]

For each generated shape the original FPSA downstream logic is reused for:
    1. cached/barycentric COACD collision-proxy transfer;
    2. task/grasp-frame transfer.

Typical usage
-------------
Reuse the object/solver/output settings from an existing FPSA meta file, but
write into a separate baseline root automatically::

    python baseline_batch_randomizer.py \
        --meta configs/bracket/fpsa_meta_bracket.yaml \
        --num-scales 16 \
        --workers 8

Or override the output root explicitly::

    python baseline_batch_randomizer.py \
        --meta fpsa_meta_bracket.yaml \
        --num-scales 16 \
        --output-root ./data/objects/bracket/uniform_scaling_baseline_outputs

The output naming stays compatible with the previous FPSA augmentation layout:
    <sample>.obj
    <sample>_coacd.obj
    <sample>_grasp.yaml
    <sample>_sample.yaml
    <sample>_debug.yaml
plus manifest.csv / manifest.jsonl at the augmentation root.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_AXIS_GROUPS: Tuple[str, ...] = ("x", "y", "z", "xy", "xz", "yz", "xyz")
VALID_AXIS_GROUPS = set(DEFAULT_AXIS_GROUPS)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


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
            path.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
    """Write the transferred grasp in the existing PickUp/FPSA loader format."""
    if not isinstance(grasp_result, dict):
        raise TypeError(f"grasp_result must be a dict, got {type(grasp_result).__name__}")

    T_mesh_hand_tcp = _matrix4(grasp_result, ["T_mesh_hand_tcp"])
    T_mesh_hand = _matrix4(grasp_result, ["T_mesh_hand"])

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


def _carry_gripper_metadata(augmentor: Any, grasp_result: Dict[str, Any]) -> None:
    """Preserve non-pose grasp metadata while leaving FPSA frame transfer untouched."""
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
# Baseline sampling helpers
# -----------------------------------------------------------------------------


def canonical_axis_group(group: str) -> str:
    """Normalize axis-combination spellings while keeping one shared scalar."""
    key = str(group).lower().strip().replace("+", "").replace(",", "").replace(" ", "")
    if not key:
        raise ValueError("Axis group cannot be empty")

    # Canonical order prevents duplicate spellings such as yx vs xy.
    chars = set(key)
    if any(c not in {"x", "y", "z"} for c in chars):
        raise ValueError(f"Invalid axis group {group!r}; use x/y/z/xy/xz/yz/xyz")
    canonical = "".join(axis for axis in "xyz" if axis in chars)
    if canonical not in VALID_AXIS_GROUPS:
        raise ValueError(f"Invalid axis group {group!r}; use x/y/z/xy/xz/yz/xyz")
    return canonical


def parse_axis_groups(value: Optional[str | Sequence[str]]) -> List[str]:
    if value is None:
        return list(DEFAULT_AXIS_GROUPS)

    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
    else:
        raw = [str(x).strip() for x in value if str(x).strip()]

    if not raw:
        raise ValueError("At least one axis group is required")

    out: List[str] = []
    seen = set()
    for item in raw:
        group = canonical_axis_group(item)
        if group not in seen:
            out.append(group)
            seen.add(group)
    return out


def scale_xyz_for_group(scale: float, axis_group: str) -> np.ndarray:
    axis_group = canonical_axis_group(axis_group)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Scale must be finite and > 0, got {scale}")

    factors = np.ones(3, dtype=np.float64)
    for axis in axis_group:
        factors[{"x": 0, "y": 1, "z": 2}[axis]] = scale
    return factors


def linspace_scales(scale_min: float, scale_max: float, count: int) -> np.ndarray:
    """Deterministic inclusive linspace, matching the old FPSA batch convention."""
    scale_min = float(scale_min)
    scale_max = float(scale_max)
    count = int(count)

    if not np.isfinite(scale_min) or not np.isfinite(scale_max):
        raise ValueError("scale_min / scale_max must be finite")
    if scale_min <= 0.0 or scale_max <= 0.0:
        raise ValueError("Scaling factors must be > 0")
    if scale_min > scale_max:
        raise ValueError(f"scale_min must be <= scale_max, got {scale_min} > {scale_max}")
    if count <= 0:
        raise ValueError(f"num_scales must be positive, got {count}")
    if count == 1:
        return np.asarray([(scale_min + scale_max) * 0.5], dtype=np.float64)
    return np.linspace(scale_min, scale_max, count, dtype=np.float64)


def _scale_tag(scale: float) -> str:
    return f"{float(scale):.6g}".replace("-", "m").replace(".", "p")


def _derive_output_root(meta: Dict[str, Any]) -> Path:
    """Keep baseline outputs separate even when reusing an FPSA meta file."""
    baseline_cfg = meta.get("baseline", {})
    output_cfg = meta.get("output", {})

    explicit = baseline_cfg.get("output_root", None)
    if explicit is None:
        explicit = output_cfg.get("baseline_root", None)
    if explicit is not None:
        return Path(explicit)

    fpsa_root = Path(output_cfg.get("root", "fpsa_aug_outputs"))
    return fpsa_root.with_name(f"{fpsa_root.name}_uniform_scaling_baseline")


# -----------------------------------------------------------------------------
# Job generation
# -----------------------------------------------------------------------------


def make_jobs(
    meta: Dict[str, Any],
    axis_groups: Optional[Sequence[str]] = None,
    num_scales: Optional[int] = None,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    seed: Optional[int] = None,
    output_root: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, Any]], Path]:
    baseline_cfg = meta.get("baseline", {})
    sampler_cfg = meta.get("sampler", {})

    groups = parse_axis_groups(
        axis_groups if axis_groups is not None else baseline_cfg.get("axis_groups", None)
    )
    n = int(
        num_scales
        if num_scales is not None
        else baseline_cfg.get("num_scales", baseline_cfg.get("n_scales_per_group", 16))
    )
    lo = float(scale_min if scale_min is not None else baseline_cfg.get("scale_min", 0.5))
    hi = float(scale_max if scale_max is not None else baseline_cfg.get("scale_max", 2.0))
    base_seed = int(seed if seed is not None else sampler_cfg.get("seed", 0))
    out_root = Path(output_root) if output_root is not None else _derive_output_root(meta)

    scales = linspace_scales(lo, hi, n)
    jobs: List[Dict[str, Any]] = []
    sample_id = 0

    # Every group independently receives the full linspace [lo, hi].
    for group in groups:
        for scale_index, scale in enumerate(scales):
            sample_seed = int(base_seed + 1000003 * sample_id)
            jobs.append(
                {
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "axis_group": group,
                    "scale_index": scale_index,
                    "num_scales": n,
                    "scale": float(scale),
                    "scale_xyz": scale_xyz_for_group(float(scale), group).tolist(),
                    "scale_min": lo,
                    "scale_max": hi,
                    "output_root": str(out_root),
                    "meta": meta,
                }
            )
            sample_id += 1

    return jobs, out_root


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------


def _worker_generate_shape(job: Dict[str, Any]) -> Dict[str, Any]:
    """Generate one baseline shape; top-level for multiprocessing spawn."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        # Import inside each spawned worker, matching the original FPSA batch rule.
        from FPSA_baseline import ShapeAugmentor

        meta = job["meta"]
        object_cfg = meta.get("object", {})
        solver_cfg = meta.get("solver", {})
        output_cfg = meta.get("output", {})

        obj_path = object_cfg.get("obj_path", None)
        if obj_path is None:
            raise KeyError("meta['object']['obj_path'] is required")
        initial_grasp_path = object_cfg.get("initial_grasp_path", None)

        stem = str(object_cfg.get("name", Path(obj_path).stem))
        group = canonical_axis_group(job["axis_group"])
        scale = float(job["scale"])
        sample_name = (
            f"{stem}_baseline_{group}_"
            f"{int(job['scale_index']):03d}_s{_scale_tag(scale)}"
        )

        out_root = Path(job["output_root"])
        # Flat is safest for direct FPSAObjectDR sampling.  Keep the old option
        # for projects already using recursive per-shape directories.
        layout = str(output_cfg.get("baseline_layout", output_cfg.get("layout", "flat")))

        if layout == "per_shape_dir":
            sample_dir = out_root / sample_name
        elif layout == "flat":
            sample_dir = out_root
        else:
            raise ValueError(f"Unknown output layout {layout!r}; use 'flat' or 'per_shape_dir'")

        obj_out = sample_dir / f"{sample_name}.obj"
        grasp_out = sample_dir / f"{sample_name}_grasp.yaml"
        meta_out = sample_dir / f"{sample_name}_sample.yaml"
        debug_out = sample_dir / f"{sample_name}_debug.yaml"

        sample_dir.mkdir(parents=True, exist_ok=True)
        overwrite = bool(output_cfg.get("baseline_overwrite", output_cfg.get("overwrite", False)))
        if obj_out.exists() and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {obj_out}")

        augmentor = ShapeAugmentor(
            obj_path=obj_path,
            initial_grasp_path=initial_grasp_path,
        )

        # Baseline-specific operation only.  One scalar is shared by every axis
        # in the selected group; unselected axes stay at factor 1.0.
        augmentor.naive_axis_uniform_scale(scale=scale, axes=group)

        # Reuse the original FPSA collision-proxy transfer unchanged.
        mesh_path, collision_path = augmentor.write_augment_obj(
            obj_out,
            write_coacd=True,
            return_paths=True,
        )

        save_grasp = bool(output_cfg.get("save_grasp", True))
        grasp_written = False
        grasp_result = None
        grasp_anchor = None
        grasp_debug = None

        if initial_grasp_path is not None and save_grasp:
            # Force dict output because PickUp's grasp loader consumes the YAML
            # fields T_mesh_hand / T_mesh_hand_tcp plus gripper metadata.
            grasp_result, grasp_anchor, grasp_debug = augmentor.transfer_initial_grasp_guess(
                k_ring=int(solver_cfg.get("k_ring", 2)),
                use_distance_weights=bool(solver_cfg.get("use_distance_weights", True)),
                quat_order=str(solver_cfg.get("quat_order", "xyzw")),
                return_format="dict",
                patch_method=str(solver_cfg.get("patch_method", "k_ring")),
            )
            _carry_gripper_metadata(augmentor, grasp_result)
            write_grasp_yaml(grasp_out, grasp_result, mesh_path=mesh_path)
            grasp_written = True

        sample_record = {
            "baseline": "naive_axis_uniform_scaling",
            "sample_id": int(job["sample_id"]),
            "sample_name": sample_name,
            "seed": int(job["seed"]),
            "axis_group": group,
            "scale_index": int(job["scale_index"]),
            "num_scales": int(job["num_scales"]),
            "scale": scale,
            "scale_xyz": list(map(float, job["scale_xyz"])),
            "scale_range": [float(job["scale_min"]), float(job["scale_max"])],
            "scale_center": "mesh_local_origin",
            "obj_path": str(mesh_path),
            "collision_path": str(collision_path),
            "grasp_path": str(grasp_out) if grasp_written else None,
            "meta_path": str(meta_out),
            "debug_path": str(debug_out),
        }
        dump_yaml_or_json(meta_out, sample_record)

        if bool(output_cfg.get("save_debug", True)):
            debug_record = {
                "sample": sample_record,
                "mesh_status": augmentor.mesh_status(),
                "grasp_anchor": grasp_anchor,
                "grasp_debug": grasp_debug,
            }
            dump_yaml_or_json(debug_out, debug_record)

        return {
            "ok": True,
            "sample_id": int(job["sample_id"]),
            "sample_name": sample_name,
            "labels": f"scale_{group}",
            "axis_group": group,
            "scale": scale,
            "scale_x": float(job["scale_xyz"][0]),
            "scale_y": float(job["scale_xyz"][1]),
            "scale_z": float(job["scale_xyz"][2]),
            "obj_path": str(mesh_path),
            "collision_path": str(collision_path),
            "grasp_path": str(grasp_out) if grasp_written else "",
            "meta_path": str(meta_out),
            "debug_path": str(debug_out) if bool(output_cfg.get("save_debug", True)) else "",
            "error": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "sample_id": int(job.get("sample_id", -1)),
            "sample_name": "",
            "labels": f"scale_{job.get('axis_group', '')}",
            "axis_group": job.get("axis_group", ""),
            "scale": job.get("scale", ""),
            "scale_x": "",
            "scale_y": "",
            "scale_z": "",
            "obj_path": "",
            "collision_path": "",
            "grasp_path": "",
            "meta_path": "",
            "debug_path": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# -----------------------------------------------------------------------------
# Manifest / runner
# -----------------------------------------------------------------------------


def write_manifest(out_root: Path, rows: List[Dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "manifest.csv"
    jsonl_path = out_root / "manifest.jsonl"

    fieldnames = [
        "ok",
        "sample_id",
        "sample_name",
        "labels",
        "axis_group",
        "scale",
        "scale_x",
        "scale_y",
        "scale_z",
        "obj_path",
        "collision_path",
        "grasp_path",
        "meta_path",
        "debug_path",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def run_batch(
    meta_path: str | Path,
    axis_groups: Optional[Sequence[str]] = None,
    num_scales: Optional[int] = None,
    workers: Optional[int] = None,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    seed: Optional[int] = None,
    output_root: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    meta = load_meta(meta_path)
    jobs, out_root = make_jobs(
        meta,
        axis_groups=axis_groups,
        num_scales=num_scales,
        scale_min=scale_min,
        scale_max=scale_max,
        seed=seed,
        output_root=output_root,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if not jobs:
        write_manifest(out_root, [])
        return []

    sampler_cfg = meta.get("sampler", {})
    requested_workers = workers if workers is not None else sampler_cfg.get("max_workers", os.cpu_count() or 1)
    max_workers = max(1, min(int(requested_workers), len(jobs)))

    ctx = mp.get_context(str(sampler_cfg.get("mp_start_method", "spawn")))
    rows: List[Dict[str, Any]] = []

    if max_workers == 1:
        for job in jobs:
            rows.append(_worker_generate_shape(job))
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = [ex.submit(_worker_generate_shape, job) for job in jobs]
            for fut in as_completed(futures):
                rows.append(fut.result())

    rows.sort(key=lambda row: int(row.get("sample_id", 10**12)))
    write_manifest(out_root, rows)

    n_ok = sum(bool(row.get("ok")) for row in rows)
    groups = parse_axis_groups(axis_groups if axis_groups is not None else meta.get("baseline", {}).get("axis_groups", None))
    print(
        f"[baseline batch] done: {n_ok}/{len(rows)} succeeded | "
        f"groups={groups} | output={out_root}"
    )
    print(f"[baseline batch] PickUp root: {out_root}")
    if n_ok != len(rows):
        print(f"[baseline batch] failed: {len(rows) - n_ok}; check {out_root / 'manifest.jsonl'}")
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate naive x/y/z axis-uniform scaling baselines with deterministic "
            "linspace sampling and FPSA collision/grasp transfer."
        )
    )
    parser.add_argument("--meta", required=True, help="Existing FPSA YAML/JSON meta file.")
    parser.add_argument(
        "--axes",
        default=None,
        help="Comma-separated groups. Default: x,y,z,xy,xz,yz,xyz",
    )
    parser.add_argument(
        "--num-scales",
        type=int,
        default=None,
        help="Linspace samples PER axis group. Default: baseline.num_scales or 16.",
    )
    parser.add_argument("--scale-min", type=float, default=None, help="Default: 0.5")
    parser.add_argument("--scale-max", type=float, default=None, help="Default: 2.0")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Baseline augmentation root. If omitted, derive a separate sibling root "
            "from output.root, e.g. fpsa_aug_outputs_uniform_scaling_baseline."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_batch(
        meta_path=args.meta,
        axis_groups=parse_axis_groups(args.axes) if args.axes is not None else None,
        num_scales=args.num_scales,
        workers=args.workers,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        seed=args.seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
