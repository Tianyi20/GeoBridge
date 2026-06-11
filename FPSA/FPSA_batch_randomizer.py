"""
Meta-driven, multiprocessing shape augmentation runner for FPSA.ShapeAugmentor.

Usage:
    python FPSA_batch_randomizer.py --meta fpsa_meta_bracket.yaml --labels bracket_x_extent,bracket_x_shrink --num-shapes 32 --workers 8

Design notes:
    - Each worker process creates its own ShapeAugmentor instance. Do not share ShapeAugmentor
      across processes because FPSA depends on C++/Open3D/libigl state.
    - Every generated shape is produced from the original mesh, not from the previous output.
    - Multiple labels can be composed in one sample by building one target_positions array and
      calling slippage_reshape once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# ----------------------------
# IO helpers
# ----------------------------


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


def write_grasp_pose_yaml(
    path: str | Path,
    T_mesh_hand_tcp: Any,
    initial_grasp_guess: Optional[Dict[str, Any]],
    mesh_path: str | Path,
) -> None:
    """Write a grasp pose YAML compatible with load_initial_grasp_pose()."""
    if isinstance(T_mesh_hand_tcp, dict):
        T = np.asarray(T_mesh_hand_tcp["T_mesh_hand_tcp"], dtype=np.float64)
    else:
        T = np.asarray(T_mesh_hand_tcp, dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError(f"T_mesh_hand_tcp must be 4x4, got {T.shape}")

    old = initial_grasp_guess if isinstance(initial_grasp_guess, dict) else {}
    opening = float(old.get("opening", old.get("opening_width_m", 0.06)))

    data = {
        "mesh_path": str(mesh_path),
        "reference_frame": str(old.get("reference_frame", "mesh_local_frame")),
        "hand_frame": str(old.get("hand_frame", "panda_hand")),
        "opening_width_m": opening,
        "pregrasp_opening_width_m": float(
            old.get("pregrasp_opening", old.get("pregrasp_opening_width_m", opening))
        ),
        "finger_joint_m": float(
            old.get("finger_joint", old.get("finger_joint_m", opening / 2.0))
        ),
        "T_mesh_hand": T,
        "T_mesh_hand_tcp": T,
    }
    dump_yaml_or_json(path, data)


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


# ----------------------------
# Meta model
# ----------------------------


@dataclass(frozen=True)
class DeformationSpec:
    label: str
    constrained_ids: List[int]
    reshaped_ids: List[int]
    reshaped_vector: Any
    range: Any
    type: str = "displacement"
    coupled: bool = True
    normalize_vector: bool = True
    distribution: str = "uniform"
    weight: float = 1.0
    max_iters: Optional[int] = None
    handle_error_distrib_enabled: Optional[bool] = None
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DeformationSpec":
        required = ["label", "constrained_ids", "reshaped_ids", "reshaped_vector", "range"]
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(f"Deformation meta missing required fields: {missing}")

        known = {
            "label", "constrained_ids", "reshaped_ids", "reshaped_vector", "range", "type",
            "coupled", "normalize_vector", "distribution", "weight", "max_iters",
            "handle_error_distrib_enabled", "description",
        }
        return DeformationSpec(
            label=str(d["label"]),
            constrained_ids=[int(v) for v in d["constrained_ids"]],
            reshaped_ids=[int(v) for v in d["reshaped_ids"]],
            reshaped_vector=d["reshaped_vector"],
            range=d["range"],
            type=str(d.get("type", "displacement")),
            coupled=bool(d.get("coupled", True)),
            normalize_vector=bool(d.get("normalize_vector", True)),
            distribution=str(d.get("distribution", "uniform")),
            weight=float(d.get("weight", 1.0)),
            max_iters=d.get("max_iters", None),
            handle_error_distrib_enabled=d.get("handle_error_distrib_enabled", None),
            description=str(d.get("description", "")),
            extra={k: v for k, v in d.items() if k not in known},
        )


def parse_deformations(meta: Dict[str, Any]) -> Dict[str, DeformationSpec]:
    deforms = meta.get("deformations", None)
    if not isinstance(deforms, list) or len(deforms) == 0:
        raise ValueError("meta['deformations'] must be a non-empty list")

    out: Dict[str, DeformationSpec] = {}
    for item in deforms:
        spec = DeformationSpec.from_dict(item)
        if spec.label in out:
            raise ValueError(f"Duplicated deformation label: {spec.label}")
        out[spec.label] = spec
    return out


# ----------------------------
# Sampling helpers
# ----------------------------


def normalize_vec(v: Sequence[float], do_normalize: bool = True) -> np.ndarray:
    v_arr = np.asarray(v, dtype=np.float64).reshape(3)
    if do_normalize:
        n = float(np.linalg.norm(v_arr))
        if n < 1e-12:
            raise ValueError(f"Zero reshaped_vector is invalid: {v}")
        v_arr = v_arr / n
    return v_arr


def vector_for_index(vec_spec: Any, i: int, n: int, normalize: bool = True) -> np.ndarray:
    """
    Accept either:
      reshaped_vector: [1, 0, 0]
      reshaped_vector: [[1,0,0], [0,1,0], ...]
      reshaped_vector: "x" / "-x" / "y" / "z"
    """
    axis = {
        "x": [1, 0, 0], "+x": [1, 0, 0], "-x": [-1, 0, 0],
        "y": [0, 1, 0], "+y": [0, 1, 0], "-y": [0, -1, 0],
        "z": [0, 0, 1], "+z": [0, 0, 1], "-z": [0, 0, -1],
    }
    if isinstance(vec_spec, str):
        key = vec_spec.lower().strip()
        if key not in axis:
            raise ValueError(f"Unknown axis string for reshaped_vector: {vec_spec}")
        return normalize_vec(axis[key], normalize)

    arr = np.asarray(vec_spec, dtype=np.float64)
    if arr.shape == (3,):
        return normalize_vec(arr, normalize)
    if arr.shape == (n, 3):
        return normalize_vec(arr[i], normalize)
    raise ValueError(f"reshaped_vector must be shape (3,) or ({n}, 3), got {arr.shape}")


def sample_one_scalar(range_spec: Any, rng: np.random.Generator, distribution: str = "uniform") -> float:
    """
    Supported range formats:
      [-0.02, 0.04]                       -> uniform low/high
      {type: uniform, low: -0.02, high: 0.04}
      {type: normal, mean: 0.0, std: 0.01, clip: [-0.02, 0.04]}
      {type: choice, values: [-0.02, 0.0, 0.04]}
      {type: fixed, value: 0.02}
    """
    if isinstance(range_spec, dict):
        typ = str(range_spec.get("type", distribution)).lower()
        if typ == "uniform":
            return float(rng.uniform(float(range_spec["low"]), float(range_spec["high"])))
        if typ == "normal":
            x = float(rng.normal(float(range_spec.get("mean", 0.0)), float(range_spec["std"])))
            if "clip" in range_spec and range_spec["clip"] is not None:
                lo, hi = range_spec["clip"]
                x = float(np.clip(x, float(lo), float(hi)))
            return x
        if typ == "choice":
            values = list(range_spec["values"])
            return float(values[int(rng.integers(0, len(values)))])
        if typ == "fixed":
            return float(range_spec["value"])
        raise ValueError(f"Unsupported range type: {typ}")

    arr = np.asarray(range_spec, dtype=np.float64)
    if arr.shape == (2,):
        lo, hi = float(arr[0]), float(arr[1])
        if distribution == "uniform":
            return float(rng.uniform(lo, hi))
        raise ValueError(f"distribution='{distribution}' needs dict range format")
    raise ValueError(f"Invalid scalar range format: {range_spec}")


def range_for_index(range_spec: Any, i: int, n: int) -> Any:
    """
    Accept either one range for all reshaped ids, or per-id ranges:
      range: [-0.02, 0.04]
      range: [[-0.02, 0.04], [-0.01, 0.03]]
      range: [{type: normal, ...}, {type: uniform, ...}]
    """
    if isinstance(range_spec, dict):
        return range_spec

    if isinstance(range_spec, list):
        # Single [low, high] range.
        if len(range_spec) == 2 and all(isinstance(v, (int, float)) for v in range_spec):
            return range_spec
        # Per-handle range list.
        if len(range_spec) == n:
            return range_spec[i]

    arr = np.asarray(range_spec, dtype=object)
    if arr.shape == (2,):
        return range_spec
    if len(range_spec) == n:
        return range_spec[i]
    raise ValueError(f"range must be one scalar range or {n} per-id ranges, got: {range_spec}")


def sample_deformation(spec: DeformationSpec, rng: np.random.Generator) -> Dict[str, Any]:
    """Sample one deformation operation from a label spec."""
    if spec.type not in {"displacement", "slippage"}:
        raise ValueError(
            f"Unsupported deformation type='{spec.type}'. "
            "Use 'displacement'/'slippage' for vector-range sampling."
        )

    n = len(spec.reshaped_ids)
    if n == 0:
        raise ValueError(f"Label {spec.label} has no reshaped_ids")

    displacements: List[List[float]] = []
    magnitudes: List[float] = []

    if spec.coupled:
        mag = sample_one_scalar(range_for_index(spec.range, 0, n), rng, spec.distribution)
        mags = [mag] * n
    else:
        mags = [sample_one_scalar(range_for_index(spec.range, i, n), rng, spec.distribution) for i in range(n)]

    for i, mag in enumerate(mags):
        direction = vector_for_index(spec.reshaped_vector, i, n, spec.normalize_vector)
        disp = float(mag) * direction
        displacements.append(disp.tolist())
        magnitudes.append(float(mag))

    # Important: automatically include handles in the constraint set, because FPSA.displacement_reshape
    # requires displace_idxs to be a subset of constraint_ids. This also matches the solver idea that
    # fixed points and handles are both constrained rows in target_positions.
    constraint_ids = unique_ints([*spec.constrained_ids, *spec.reshaped_ids])

    return {
        "label": spec.label,
        "type": spec.type,
        "constraint_ids": constraint_ids,
        "reshaped_ids": [int(v) for v in spec.reshaped_ids],
        "displacements": displacements,
        "magnitudes": magnitudes,
        "description": spec.description,
    }


def unique_ints(xs: Iterable[int]) -> List[int]:
    seen = set()
    out = []
    for x in xs:
        x = int(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def choose_labels(
    all_specs: Dict[str, DeformationSpec],
    selected_labels: Optional[Sequence[str]],
    n_shapes: int,
    mode: str,
    rng: np.random.Generator,
    labels_per_shape: int = 1,
) -> List[List[str]]:
    labels = list(selected_labels) if selected_labels else list(all_specs.keys())
    unknown = [x for x in labels if x not in all_specs]
    if unknown:
        raise KeyError(f"Unknown deformation labels: {unknown}. Available: {list(all_specs.keys())}")
    if n_shapes <= 0:
        return []

    mode = mode.lower()
    labels_per_shape = max(1, int(labels_per_shape))

    if mode == "balanced":
        jobs = []
        for i in range(n_shapes):
            jobs.append([labels[i % len(labels)]])
        rng.shuffle(jobs)
        return jobs

    if mode == "random_one":
        weights = np.asarray([max(0.0, all_specs[l].weight) for l in labels], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(labels), dtype=np.float64)
        probs = weights / weights.sum()
        idxs = rng.choice(len(labels), size=n_shapes, replace=True, p=probs)
        return [[labels[int(i)]] for i in idxs]

    if mode == "random_k":
        k = min(labels_per_shape, len(labels))
        return [list(rng.choice(labels, size=k, replace=False)) for _ in range(n_shapes)]

    if mode == "all":
        # Each shape composes all selected labels in one solve.
        return [list(labels) for _ in range(n_shapes)]

    raise ValueError("label_mode must be one of: balanced, random_one, random_k, all")


# ----------------------------
# Worker and runner
# ----------------------------


def build_target_positions(augmentor: Any, sampled_ops: List[Dict[str, Any]]) -> Tuple[List[int], np.ndarray]:
    """
    Compose one or more sampled label operations into a single target_positions array.
    The base is always augmentor.V from the original mesh.
    """
    all_constraint_ids = unique_ints(v for op in sampled_ops for v in op["constraint_ids"])
    row_of = {vid: i for i, vid in enumerate(all_constraint_ids)}
    target_positions = np.asarray(augmentor.V[all_constraint_ids], dtype=np.float64).copy()

    for op in sampled_ops:
        for vid, disp in zip(op["reshaped_ids"], op["displacements"]):
            row = row_of[int(vid)]
            target_positions[row] += np.asarray(disp, dtype=np.float64).reshape(3)

    return all_constraint_ids, target_positions


def _worker_generate_shape(job: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level function so it is picklable under spawn."""
    # Avoid oversubscription: many C++ numeric libraries use their own thread pools.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        # Import inside the process to avoid sharing C++ extension state from parent.
        from FPSA import ShapeAugmentor

        meta = job["meta"]
        sampled_ops = job["sampled_ops"]
        object_cfg = meta.get("object", {})
        solver_cfg = meta.get("solver", {})
        output_cfg = meta.get("output", {})

        obj_path = object_cfg["obj_path"]
        initial_grasp_path = object_cfg.get("initial_grasp_path", None)
        auto_repair = bool(object_cfg.get("auto_repair", True))

        stem = object_cfg.get("name", Path(obj_path).stem)
        sample_name = f"{stem}_{job['sample_id']:06d}"
        out_root = Path(output_cfg.get("root", "fpsa_aug_outputs"))
        layout = output_cfg.get("layout", "flat")

        if layout == "per_shape_dir":
            sample_dir = out_root / sample_name
            obj_out = sample_dir / f"{sample_name}.obj"
            grasp_out = sample_dir / f"{sample_name}_grasp.yaml"
            meta_out = sample_dir / f"{sample_name}_sample.yaml"
            debug_out = sample_dir / f"{sample_name}_debug.yaml"
        else:
            sample_dir = out_root
            obj_out = out_root / f"{sample_name}.obj"
            grasp_out = out_root / f"{sample_name}_grasp.yaml"
            meta_out = out_root / f"{sample_name}_sample.yaml"
            debug_out = out_root / f"{sample_name}_debug.yaml"

        sample_dir.mkdir(parents=True, exist_ok=True)
        if obj_out.exists() and not bool(output_cfg.get("overwrite", False)):
            raise FileExistsError(f"Output exists and overwrite=false: {obj_out}")

        augmentor = ShapeAugmentor(
            obj_path=obj_path,
            initial_grasp_path=initial_grasp_path,
            auto_repair=auto_repair,
        )

        constraint_ids, target_positions = build_target_positions(augmentor, sampled_ops)

        # Per-label max_iters can override solver only when a single label is used;
        # for composed labels, use global solver max_iters to keep behavior predictable.
        max_iters = int(solver_cfg.get("max_iters", 20))
        handle_error = bool(solver_cfg.get("handle_error_distrib_enabled", False))
        visualize_grasp_transfer = bool(solver_cfg.get("visualize_grasp_transfer", False))
        if len(sampled_ops) == 1:
            label_name = sampled_ops[0]["label"]
            spec_dict = {d["label"]: d for d in meta.get("deformations", [])}[label_name]
            if spec_dict.get("max_iters", None) is not None:
                max_iters = int(spec_dict["max_iters"])
            if spec_dict.get("handle_error_distrib_enabled", None) is not None:
                handle_error = bool(spec_dict["handle_error_distrib_enabled"])

        augmentor.slippage_reshape(
            constraint_ids=constraint_ids,
            target_positions=target_positions,
            max_iters=max_iters,
            handle_error_distrib_enabled=handle_error,
            input_name=sample_name,
        )
        augmentor.write_augment_obj(obj_out)

        grasp_written = False
        grasp_debug = None
        if initial_grasp_path is not None and bool(output_cfg.get("save_grasp", True)):
            grasp_result, anchor, grasp_debug = augmentor.transfer_initial_grasp_guess(
                k_ring=int(solver_cfg.get("k_ring", 2)),
                use_distance_weights=bool(solver_cfg.get("use_distance_weights", True)),
                quat_order=str(solver_cfg.get("quat_order", "xyzw")),
                return_format=str(solver_cfg.get("return_format", "dict")),
            )
            if visualize_grasp_transfer:
                augmentor.visualize_deformed_grasp_pose(
                            T_grasp_new=grasp_result,
                            anchor=anchor,
                            debug_info=grasp_debug,
                            show_anchor=True,
                            show_patch=True,
                        )
            write_grasp_pose_yaml(
                grasp_out,
                T_mesh_hand_tcp=grasp_result,
                initial_grasp_guess=augmentor.initial_grasp_guess,
                mesh_path=obj_out,
            )
            grasp_written = True
        else:
            anchor = None

        sample_record = {
            "sample_id": int(job["sample_id"]),
            "sample_name": sample_name,
            "seed": int(job["seed"]),
            "labels": [op["label"] for op in sampled_ops],
            "sampled_ops": sampled_ops,
            "constraint_ids": constraint_ids,
            "obj_path": str(obj_out),
            "grasp_path": str(grasp_out) if grasp_written else None,
            "meta_path": str(meta_out),
            "debug_path": str(debug_out),
        }
        dump_yaml_or_json(meta_out, sample_record)

        if bool(output_cfg.get("save_debug", True)):
            debug_record = {
                "sample": sample_record,
                "mesh_status": augmentor.mesh_status(),
                "grasp_anchor": anchor,
                "grasp_debug": grasp_debug,
            }
            dump_yaml_or_json(debug_out, debug_record)

        return {
            "ok": True,
            "sample_id": int(job["sample_id"]),
            "sample_name": sample_name,
            "labels": "+".join([op["label"] for op in sampled_ops]),
            "obj_path": str(obj_out),
            "grasp_path": str(grasp_out) if grasp_written else "",
            "meta_path": str(meta_out),
            "debug_path": str(debug_out) if bool(output_cfg.get("save_debug", True)) else "",
            "error": "",
        }

    except Exception as e:  # return error instead of killing whole batch
        return {
            "ok": False,
            "sample_id": int(job.get("sample_id", -1)),
            "sample_name": job.get("sample_name", ""),
            "labels": "+".join(job.get("label_names", [])),
            "obj_path": "",
            "grasp_path": "",
            "meta_path": "",
            "debug_path": "",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def make_jobs(
    meta: Dict[str, Any],
    labels: Optional[Sequence[str]] = None,
    num_shapes: Optional[int] = None,
    seed: Optional[int] = None,
    label_mode: Optional[str] = None,
    labels_per_shape: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sampler_cfg = meta.get("sampler", {})
    n_shapes = int(num_shapes if num_shapes is not None else sampler_cfg.get("n_shapes", 1))
    base_seed = int(seed if seed is not None else sampler_cfg.get("seed", 0))
    mode = str(label_mode if label_mode is not None else sampler_cfg.get("label_mode", "balanced"))
    k = int(labels_per_shape if labels_per_shape is not None else sampler_cfg.get("labels_per_shape", 1))

    all_specs = parse_deformations(meta)
    rng = np.random.default_rng(base_seed)
    selected = list(labels) if labels else sampler_cfg.get("labels", None)
    schedule = choose_labels(all_specs, selected, n_shapes, mode, rng, labels_per_shape=k)

    jobs = []
    for sample_id, label_names in enumerate(schedule):
        sample_seed = int(base_seed + 1000003 * sample_id)
        local_rng = np.random.default_rng(sample_seed)
        sampled_ops = [sample_deformation(all_specs[label], local_rng) for label in label_names]
        # print(sampled_ops[0]["displacements"])
        jobs.append({
            "sample_id": sample_id,
            "seed": sample_seed,
            "label_names": label_names,
            "sampled_ops": sampled_ops,
            "meta": meta,
        })
    return jobs


def write_manifest(out_root: Path, rows: List[Dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "manifest.csv"
    jsonl_path = out_root / "manifest.jsonl"

    fieldnames = ["ok", "sample_id", "sample_name", "labels", "obj_path", "grasp_path", "meta_path", "debug_path", "error"]
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
    labels: Optional[Sequence[str]] = None,
    num_shapes: Optional[int] = None,
    workers: Optional[int] = None,
    seed: Optional[int] = None,
    label_mode: Optional[str] = None,
    labels_per_shape: Optional[int] = None,
) -> List[Dict[str, Any]]:
    meta = load_meta(meta_path)
    out_root = Path(meta.get("output", {}).get("root", "fpsa_aug_outputs"))
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(
        meta,
        labels=labels,
        num_shapes=num_shapes,
        seed=seed,
        label_mode=label_mode,
        labels_per_shape=labels_per_shape,
    )
    if not jobs:
        return []

    sampler_cfg = meta.get("sampler", {})
    max_workers = int(workers if workers is not None else sampler_cfg.get("max_workers", max(1, os.cpu_count() or 1)))
    max_workers = max(1, min(max_workers, len(jobs)))

    # spawn is safer than fork for C++ extensions / Open3D / libigl.
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

    rows = sorted(rows, key=lambda x: int(x.get("sample_id", 10**12)))
    write_manifest(out_root, rows)

    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"[FPSA batch] done: {n_ok}/{len(rows)} succeeded. manifest: {out_root / 'manifest.csv'}")
    if n_ok != len(rows):
        print(f"[FPSA batch] failed: {len(rows) - n_ok}; check manifest.jsonl for tracebacks")
    return rows


def _split_labels(s: Optional[str]) -> Optional[List[str]]:
    if s is None or s.strip() == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to YAML/JSON meta file")
    parser.add_argument("--labels", default=None, help="Comma-separated labels to use; overrides sampler.labels")
    parser.add_argument("--num-shapes", type=int, default=None, help="Number of output shapes; overrides sampler.n_shapes")
    parser.add_argument("--workers", type=int, default=None, help="Number of processes; overrides sampler.max_workers")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed; overrides sampler.seed")
    parser.add_argument("--label-mode", default=None, choices=["balanced", "random_one", "random_k", "all"])
    parser.add_argument("--labels-per-shape", type=int, default=None, help="Used only by label-mode=random_k")
    args = parser.parse_args()

    run_batch(
        meta_path=args.meta,
        labels=_split_labels(args.labels),
        num_shapes=args.num_shapes,
        workers=args.workers,
        seed=args.seed,
        label_mode=args.label_mode,
        labels_per_shape=args.labels_per_shape,
    )


if __name__ == "__main__":
    main()
