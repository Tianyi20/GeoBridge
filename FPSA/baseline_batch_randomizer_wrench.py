"""Batch generator for the naive axis-uniform scaling wrench baseline.

This version follows the same frame-normalization pipeline as chain_demo.py.
For every generated baseline shape:

    1. apply naive axis-uniform scaling;
    2. transfer the original mesh-origin frame (identity SE(3)) through the
       deformation to obtain ``T_new``;
    3. transform the deformed mesh by ``inv(T_new)`` so the transferred origin
       becomes the new mesh origin;
    4. export the transformed mesh and its FPSA-transferred COACD collision proxy;
    5. compute exactly as chain_demo.py::

           wrench_to_tcp_T = T_init_tcp @ inv(T_new)

       and save it as ``<sample>_wrench_to_tcp.yaml`` for WrenchSim.

The deformation schedule stays the same as the previous baseline randomizer.
A shared scalar ``s`` is sampled by deterministic ``np.linspace`` over
``[scale_min, scale_max]`` independently for every selected axis group:

    x   -> [s, 1, 1]
    y   -> [1, s, 1]
    z   -> [1, 1, s]
    xy  -> [s, s, 1]
    xz  -> [s, 1, s]
    yz  -> [1, s, s]
    xyz -> [s, s, s]

Default ``T_init_tcp`` matches chain_demo.py: identity rotation with translation
``[0.06989, 0, 0]``.  It can be overridden in the meta file with either::

    baseline:
      init_tcp_translation: [0.06989, 0.0, 0.0]

or a full 4x4 matrix::

    baseline:
      T_init_tcp: [[...], [...], [...], [...]]

Typical usage::

    python baseline_batch_randomizer_wrench.py \
        --meta configs/wrench/wrench_quality_stretch.yaml \
        --num-scales 16 \
        --workers 8 \
        --scale-min 0.5 \
        --scale-max 2.0
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


def build_T_init_tcp(meta: Dict[str, Any]) -> np.ndarray:
    """Build the fixed TCP transform used by the chain_demo convention."""
    baseline_cfg = meta.get("baseline", {})
    object_cfg = meta.get("object", {})

    matrix_value = _first_present(
        baseline_cfg,
        ["T_init_tcp", "t_init_tcp", "initial_tcp_T"],
        default=_first_present(object_cfg, ["T_init_tcp", "initial_tcp_T"], None),
    )
    if matrix_value is not None:
        T = np.asarray(matrix_value, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_init_tcp must be 4x4, got {T.shape}")
        return T.copy()

    translation = _first_present(
        baseline_cfg,
        ["init_tcp_translation", "initial_tcp_translation", "tcp_translation"],
        default=_first_present(
            object_cfg,
            ["init_tcp_translation", "initial_tcp_translation", "tcp_translation"],
            [0.06989, 0.0, 0.0],
        ),
    )
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = translation
    return T


def write_wrench_to_tcp_yaml(
    path: str | Path,
    *,
    wrench_to_tcp_T: np.ndarray,
    T_new: np.ndarray,
    T_init_tcp: np.ndarray,
    mesh_path: str | Path,
    collision_path: str | Path,
    axis_group: str,
    scale: float,
    scale_xyz: Sequence[float],
) -> None:
    """Save the final normalized-wrench -> fixed-TCP transform for WrenchSim."""
    record = {
        "mesh_path": str(mesh_path),
        "collision_path": str(collision_path),
        "reference_frame": "normalized_wrench_mesh_origin",
        "target_frame": "fixed_tcp",
        "wrench_to_tcp_T": np.asarray(wrench_to_tcp_T, dtype=np.float64),
        # Useful provenance/debug fields.  The main downstream key is wrench_to_tcp_T.
        "T_new": np.asarray(T_new, dtype=np.float64),
        "T_init_tcp": np.asarray(T_init_tcp, dtype=np.float64),
        "axis_group": str(axis_group),
        "scale": float(scale),
        "scale_xyz": np.asarray(scale_xyz, dtype=np.float64),
    }
    dump_yaml_or_json(path, record)




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
    """Generate one normalized wrench baseline sample; top-level for spawn."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        # Prefer the renamed baseline module to avoid stale import/cache issues.
        try:
            from FPSA_baseline_v2 import ShapeAugmentor
        except ImportError:
            from FPSA_baseline import ShapeAugmentor

        meta = job["meta"]
        object_cfg = meta.get("object", {})
        solver_cfg = meta.get("solver", {})
        output_cfg = meta.get("output", {})

        obj_path = object_cfg.get("obj_path", None)
        if obj_path is None:
            raise KeyError("meta['object']['obj_path'] is required")

        stem = str(object_cfg.get("name", Path(obj_path).stem))
        group = canonical_axis_group(job["axis_group"])
        scale = float(job["scale"])
        sample_name = (
            f"{stem}_baseline_{group}_"
            f"{int(job['scale_index']):03d}_s{_scale_tag(scale)}"
        )

        out_root = Path(job["output_root"])
        layout = str(output_cfg.get("baseline_layout", output_cfg.get("layout", "flat")))
        if layout == "per_shape_dir":
            sample_dir = out_root / sample_name
        elif layout == "flat":
            sample_dir = out_root
        else:
            raise ValueError(f"Unknown output layout {layout!r}; use 'flat' or 'per_shape_dir'")

        obj_out = sample_dir / f"{sample_name}.obj"
        wrench_to_tcp_out = sample_dir / f"{sample_name}_wrench_to_tcp.yaml"
        meta_out = sample_dir / f"{sample_name}_sample.yaml"
        debug_out = sample_dir / f"{sample_name}_debug.yaml"

        sample_dir.mkdir(parents=True, exist_ok=True)
        overwrite = bool(output_cfg.get("baseline_overwrite", output_cfg.get("overwrite", False)))
        if obj_out.exists() and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {obj_out}")

        # No initial grasp is required here: the transferred frame is the original
        # mesh origin itself, exactly like chain_demo.py.
        augmentor = ShapeAugmentor(obj_path=obj_path, initial_grasp_path=None)

        # ------------------------------------------------------------------
        # Step 1: baseline shape deformation = naive axis-uniform rescaling.
        # ------------------------------------------------------------------
        augmentor.naive_axis_uniform_scale(scale=scale, axes=group)

        # ------------------------------------------------------------------
        # Step 2: transfer the ORIGINAL mesh-origin task frame through the
        # deformation.  T_origin is identity, same convention as chain_demo.
        # ------------------------------------------------------------------
        T_origin = np.eye(4, dtype=np.float64)
        T_new, origin_anchor, origin_debug = augmentor.transfer_grasp_SE3(
            T_grasp_old=T_origin,
            k_ring=int(solver_cfg.get("k_ring", 3)),
            use_distance_weights=bool(solver_cfg.get("use_distance_weights", True)),
            quat_order=str(solver_cfg.get("quat_order", "xyzw")),
            patch_method=str(solver_cfg.get("patch_method", "k_ring")),
            num_patch_vertices=int(solver_cfg.get("num_patch_vertices", 32)),
        )

        # ------------------------------------------------------------------
        # Step 3: move the transferred origin back to identity.
        # This mirrors chain_demo exactly:
        #     augmentor.apply_transformation_to_mesh(inv(T_new))
        # ------------------------------------------------------------------
        T_new_inv = np.linalg.inv(T_new)
        augmentor.apply_transformation_to_mesh(T_new_inv)

        # Export ONLY after origin normalization, so both the visual mesh and
        # cached/barycentric COACD proxy live in the normalized wrench frame.
        mesh_path, collision_path = augmentor.write_augment_obj(
            output_path=obj_out,
            write_coacd=True,
            return_paths=True,
        )

        # ------------------------------------------------------------------
        # Step 4: fixed TCP transform, exactly the chain_demo formula.
        # ------------------------------------------------------------------
        T_init_tcp = build_T_init_tcp(meta)
        wrench_to_tcp_T = T_init_tcp @ T_new_inv

        write_wrench_to_tcp_yaml(
            wrench_to_tcp_out,
            wrench_to_tcp_T=wrench_to_tcp_T,
            T_new=T_new,
            T_init_tcp=T_init_tcp,
            mesh_path=mesh_path,
            collision_path=collision_path,
            axis_group=group,
            scale=scale,
            scale_xyz=job["scale_xyz"],
        )

        sample_record = {
            "baseline": "naive_axis_uniform_scaling_wrench_origin_transfer",
            "sample_id": int(job["sample_id"]),
            "sample_name": sample_name,
            "seed": int(job["seed"]),
            "axis_group": group,
            "scale_index": int(job["scale_index"]),
            "num_scales": int(job["num_scales"]),
            "scale": scale,
            "scale_xyz": list(map(float, job["scale_xyz"])),
            "scale_range": [float(job["scale_min"]), float(job["scale_max"])],
            "deformation_center": "original_mesh_local_origin",
            "final_frame": "transferred_origin_normalized_to_identity",
            "obj_path": str(mesh_path),
            "collision_path": str(collision_path),
            "wrench_to_tcp_path": str(wrench_to_tcp_out),
            "T_new": T_new,
            "T_new_inv": T_new_inv,
            "T_init_tcp": T_init_tcp,
            "wrench_to_tcp_T": wrench_to_tcp_T,
            "meta_path": str(meta_out),
            "debug_path": str(debug_out),
        }
        dump_yaml_or_json(meta_out, sample_record)

        if bool(output_cfg.get("save_debug", True)):
            debug_record = {
                "sample": sample_record,
                "origin_anchor": origin_anchor,
                "origin_transfer_debug": origin_debug,
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
            "wrench_to_tcp_path": str(wrench_to_tcp_out),
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
            "wrench_to_tcp_path": "",
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
        "wrench_to_tcp_path",
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
    print(f"[baseline batch] WrenchSim root: {out_root}")
    if n_ok != len(rows):
        print(f"[baseline batch] failed: {len(rows) - n_ok}; check {out_root / 'manifest.jsonl'}")
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate naive axis-uniform scaling wrench baselines with deterministic "
            "linspace sampling, origin-frame transfer/normalization, COACD transfer, "
            "and wrench_to_tcp_T YAML export."
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
