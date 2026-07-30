"""Multiprocessing randomizer for constrained FPSA gear deformation.

This runner is intentionally single-stage:

1. Read one deformation label from ``gear_streching.yaml``.
2. Sample all gear handles together for one output shape (paired sampling, never a
   Cartesian product across handles).
3. Run exactly one slippage deformation through ``GearHoleHardConstraint``.
4. Optionally enforce the center-hole vertices exactly after the solver.
5. Export the visual mesh, cached COACD mesh, transferred grasp YAML, sample
   metadata, debug data, and batch manifests.

The existing ``gear_streching.yaml`` works without modification.  If it does not
contain a ``hole_constraint`` section, the defaults from ``gear_demo.py`` are used:
axis=z, center=[0,0,0], radius=0.0075, radial_tolerance=0.002,
target_mode=original, component_mode=all, post_enforce=true.

Examples
--------
Dry-run sampling only (does not import FPSA/Open3D/libigl):

    python FPSA_gear_randomizer.py --meta gear_streching.yaml --dry-run

Generate the YAML-configured batch:

    python FPSA_gear_randomizer.py --meta gear_streching.yaml

Override common batch settings:

    python FPSA_gear_randomizer.py \
        --meta gear_streching.yaml \
        --num-shapes 100 \
        --workers 24 \
        --output-root Gear_aug_outputs
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import re
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# The current gear demo values.  A YAML ``hole_constraint`` section overrides
# any of these values.
DEFAULT_HOLE_CONSTRAINT: Dict[str, Any] = {
    "axis": "z",
    "center": [0.0, 0.0, 0.0],
    "radius": 0.0075,
    "radial_tolerance": 0.003,
    "min_vertices": 8,
    "target_mode": "original",
    "component_mode": "all",
    "post_enforce": True,
}


# -----------------------------------------------------------------------------
# Generic IO helpers
# -----------------------------------------------------------------------------


def load_meta(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required. Install it with: pip install pyyaml")
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported meta suffix: {path.suffix}; use YAML or JSON")

    if not isinstance(data, dict):
        raise TypeError(f"Meta root must be a mapping, got {type(data).__name__}")
    return data


def to_builtin(value: Any) -> Any:
    """Convert common scientific-Python objects to YAML/JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if is_dataclass(value):
        return to_builtin(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return to_builtin(value.item())
        except Exception:
            pass
    return repr(value)


def dump_yaml_or_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_builtin(data)

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            path = path.with_suffix(".json")
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    else:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return path


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_.")
    return cleaned or "sample"


def unique_ints(values: Iterable[int]) -> List[int]:
    seen: set[int] = set()
    result: List[int] = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def matrix4(value: Any, name: str) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got shape {mat.shape}")
    return mat


# -----------------------------------------------------------------------------
# Meta model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeformationSpec:
    label: str
    constrained_ids: List[int]
    reshaped_ids: List[int]
    reshaped_vector: Any
    range: Any
    type: str = "displacement"
    method: str = "slippage"
    coupled: bool = True
    normalize_vector: bool = True
    distribution: str = "uniform"
    weight: float = 1.0
    max_iters: Optional[int] = None
    handle_error_distrib_enabled: Optional[bool] = None
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "DeformationSpec":
        required = [
            "label",
            "constrained_ids",
            "reshaped_ids",
            "reshaped_vector",
            "range",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise KeyError(f"Deformation meta missing required fields: {missing}")

        typ = str(raw.get("type", "displacement")).lower().strip()
        method = str(raw.get("method", "slippage")).lower().strip()
        if typ == "chain":
            raise ValueError(
                f"Deformation '{raw.get('label')}' is a chain. "
                "FPSA_gear_randomizer only accepts one primitive slippage deformation."
            )
        if typ not in {"displacement", "slippage"}:
            raise ValueError(
                f"Unsupported deformation type={typ!r}; use 'displacement' or 'slippage'"
            )
        if method != "slippage":
            raise ValueError(
                f"Unsupported method={method!r}; FPSA_gear_randomizer uses slippage only"
            )

        known = {
            "label",
            "constrained_ids",
            "reshaped_ids",
            "reshaped_vector",
            "range",
            "type",
            "method",
            "coupled",
            "normalize_vector",
            "distribution",
            "weight",
            "max_iters",
            "handle_error_distrib_enabled",
            "description",
        }
        return DeformationSpec(
            label=str(raw["label"]),
            constrained_ids=[int(v) for v in raw["constrained_ids"]],
            reshaped_ids=[int(v) for v in raw["reshaped_ids"]],
            reshaped_vector=raw["reshaped_vector"],
            range=raw["range"],
            type=typ,
            method=method,
            coupled=bool(raw.get("coupled", True)),
            normalize_vector=bool(raw.get("normalize_vector", True)),
            distribution=str(raw.get("distribution", "uniform")),
            weight=float(raw.get("weight", 1.0)),
            max_iters=(
                int(raw["max_iters"])
                if raw.get("max_iters") is not None
                else None
            ),
            handle_error_distrib_enabled=(
                bool(raw["handle_error_distrib_enabled"])
                if raw.get("handle_error_distrib_enabled") is not None
                else None
            ),
            description=str(raw.get("description", "")),
            extra={key: value for key, value in raw.items() if key not in known},
        )


def parse_deformations(meta: Dict[str, Any]) -> Dict[str, DeformationSpec]:
    raw_deformations = meta.get("deformations")
    if not isinstance(raw_deformations, list) or not raw_deformations:
        raise ValueError("meta['deformations'] must be a non-empty list")

    result: Dict[str, DeformationSpec] = {}
    for raw in raw_deformations:
        if not isinstance(raw, dict):
            raise TypeError("Every deformation entry must be a mapping")
        spec = DeformationSpec.from_dict(raw)
        if spec.label in result:
            raise ValueError(f"Duplicated deformation label: {spec.label}")
        if not spec.reshaped_ids:
            raise ValueError(f"Deformation '{spec.label}' has no reshaped_ids")
        result[spec.label] = spec
    return result


def resolve_hole_constraint_config(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Read optional hole settings while keeping the current YAML compatible."""
    raw: Any = None
    for key in ("hole_constraint", "gear_hole_constraint", "constraint"):
        candidate = meta.get(key)
        if isinstance(candidate, dict):
            raw = candidate
            break

    cfg = copy.deepcopy(DEFAULT_HOLE_CONSTRAINT)
    if raw is not None:
        cfg.update(raw)

    if not bool(cfg.pop("enabled", True)):
        raise ValueError(
            "The center-hole constraint cannot be disabled in FPSA_gear_randomizer"
        )

    allowed = {
        "axis",
        "center",
        "radius",
        "radial_tolerance",
        "min_vertices",
        "target_mode",
        "component_mode",
        "post_enforce",
    }
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise KeyError(f"Unknown hole_constraint keys: {unknown}")

    cfg["min_vertices"] = int(cfg["min_vertices"])
    cfg["target_mode"] = str(cfg["target_mode"]).lower()
    cfg["component_mode"] = str(cfg["component_mode"]).lower()
    cfg["post_enforce"] = bool(cfg["post_enforce"])
    return cfg


def resolve_obj_path(meta: Dict[str, Any]) -> str:
    object_cfg = meta.get("object", {})
    if not isinstance(object_cfg, dict) or not object_cfg.get("obj_path"):
        raise KeyError("meta.object.obj_path is required")
    return str(object_cfg["obj_path"])


def resolve_initial_grasp_path(meta: Dict[str, Any]) -> Optional[str]:
    value = meta.get("object", {}).get("initial_grasp_path")
    return None if value in {None, ""} else str(value)


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------


def normalize_vec(value: Sequence[float], enabled: bool) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64).reshape(3)
    if enabled:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            raise ValueError(f"Zero reshaped_vector is invalid: {value}")
        vec = vec / norm
    return vec


def vector_for_index(
    vector_spec: Any,
    index: int,
    count: int,
    normalize: bool,
) -> np.ndarray:
    axes = {
        "x": [1.0, 0.0, 0.0],
        "+x": [1.0, 0.0, 0.0],
        "-x": [-1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "+y": [0.0, 1.0, 0.0],
        "-y": [0.0, -1.0, 0.0],
        "z": [0.0, 0.0, 1.0],
        "+z": [0.0, 0.0, 1.0],
        "-z": [0.0, 0.0, -1.0],
    }
    if isinstance(vector_spec, str):
        key = vector_spec.lower().strip()
        if key not in axes:
            raise ValueError(f"Unknown reshaped_vector axis: {vector_spec}")
        return normalize_vec(axes[key], normalize)

    arr = np.asarray(vector_spec, dtype=np.float64)
    if arr.shape == (3,):
        return normalize_vec(arr, normalize)
    if arr.shape == (1, 3):
        return normalize_vec(arr[0], normalize)
    if arr.shape == (count, 3):
        return normalize_vec(arr[index], normalize)
    raise ValueError(
        f"reshaped_vector must have shape (3,), (1,3), or ({count},3); got {arr.shape}"
    )


def range_for_index(range_spec: Any, index: int, count: int) -> Any:
    if isinstance(range_spec, dict):
        return range_spec

    if isinstance(range_spec, (list, tuple)):
        values = list(range_spec)
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            return values
        if len(values) == 1:
            return values[0]
        if len(values) == count:
            return values[index]

    arr = np.asarray(range_spec, dtype=object)
    if arr.shape == (2,):
        return range_spec
    raise ValueError(
        f"range must be [low, high], one broadcast range, or {count} per-id ranges; "
        f"got {range_spec}"
    )


def scalar_from_range(
    range_spec: Any,
    *,
    sampling_mode: str,
    linspace_index: int,
    linspace_count: int,
    rng: np.random.Generator,
    distribution: str,
) -> float:
    sampling_mode = str(sampling_mode).lower()
    if sampling_mode not in {"linspace", "random"}:
        raise ValueError("sampler.value_sampling must be 'linspace' or 'random'")
    if linspace_count <= 0:
        raise ValueError("linspace_count must be positive")

    def uniform_value(low: float, high: float) -> float:
        low = float(low)
        high = float(high)
        if sampling_mode == "random":
            return float(rng.uniform(low, high))
        if linspace_count == 1:
            return float(0.5 * (low + high))
        return float(
            np.linspace(low, high, linspace_count, dtype=np.float64)[linspace_index]
        )

    if isinstance(range_spec, dict):
        typ = str(range_spec.get("type", distribution)).lower()
        if typ == "uniform":
            return uniform_value(range_spec["low"], range_spec["high"])
        if typ == "fixed":
            return float(range_spec["value"])
        if typ == "choice":
            values = [float(v) for v in range_spec["values"]]
            if not values:
                raise ValueError("choice range requires a non-empty values list")
            if sampling_mode == "random":
                return float(rng.choice(values))
            return values[linspace_index % len(values)]
        if typ == "normal":
            if sampling_mode != "random":
                raise ValueError(
                    "normal sampling requires sampler.value_sampling='random'"
                )
            value = float(
                rng.normal(float(range_spec["mean"]), float(range_spec["std"]))
            )
            clip = range_spec.get("clip")
            if clip is not None:
                low, high = map(float, clip)
                value = float(np.clip(value, low, high))
            return value
        raise ValueError(f"Unsupported range type: {typ}")

    arr = np.asarray(range_spec, dtype=np.float64)
    if arr.shape == (2,):
        return uniform_value(float(arr[0]), float(arr[1]))
    raise ValueError(f"Invalid scalar range format: {range_spec}")


def sample_deformation(
    spec: DeformationSpec,
    *,
    rng: np.random.Generator,
    sampling_mode: str,
    linspace_index: int,
    linspace_count: int,
) -> Dict[str, Any]:
    """Sample all handles for one shape without a Cartesian product."""
    count = len(spec.reshaped_ids)

    if spec.coupled:
        magnitude = scalar_from_range(
            range_for_index(spec.range, 0, count),
            sampling_mode=sampling_mode,
            linspace_index=linspace_index,
            linspace_count=linspace_count,
            rng=rng,
            distribution=spec.distribution,
        )
        magnitudes = [magnitude] * count
    else:
        # Paired sampling: every handle uses the same sample index.  This creates
        # exactly linspace_count shapes, not linspace_count ** num_handles.
        magnitudes = [
            scalar_from_range(
                range_for_index(spec.range, handle_index, count),
                sampling_mode=sampling_mode,
                linspace_index=linspace_index,
                linspace_count=linspace_count,
                rng=rng,
                distribution=spec.distribution,
            )
            for handle_index in range(count)
        ]

    displacements: List[List[float]] = []
    for handle_index, magnitude in enumerate(magnitudes):
        direction = vector_for_index(
            spec.reshaped_vector,
            handle_index,
            count,
            spec.normalize_vector,
        )
        displacements.append((float(magnitude) * direction).tolist())

    constraint_ids = unique_ints([*spec.constrained_ids, *spec.reshaped_ids])
    return {
        "label": spec.label,
        "method": "slippage",
        "constraint_ids": constraint_ids,
        "reshaped_ids": [int(v) for v in spec.reshaped_ids],
        "displacements": displacements,
        "magnitudes": [float(v) for v in magnitudes],
        "linspace_index": int(linspace_index),
        "linspace_count": int(linspace_count),
        "sampling_mode": sampling_mode,
        "sampling_strategy": "paired",
        "description": spec.description,
    }


def choose_labels(
    specs: Dict[str, DeformationSpec],
    selected_labels: Optional[Sequence[str]],
    n_shapes: int,
    mode: str,
    rng: np.random.Generator,
) -> List[str]:
    labels = list(selected_labels) if selected_labels else list(specs)
    unknown = [label for label in labels if label not in specs]
    if unknown:
        raise KeyError(f"Unknown labels: {unknown}. Available: {list(specs)}")
    if not labels:
        raise ValueError("No deformation labels were selected")
    if n_shapes <= 0:
        return []

    mode = str(mode).lower()
    if mode == "balanced":
        schedule = [labels[index % len(labels)] for index in range(n_shapes)]
        rng.shuffle(schedule)
        return schedule

    if mode == "random_one":
        weights = np.asarray(
            [max(0.0, specs[label].weight) for label in labels],
            dtype=np.float64,
        )
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(labels), dtype=np.float64)
        probabilities = weights / weights.sum()
        selected = rng.choice(labels, size=n_shapes, replace=True, p=probabilities)
        return [str(label) for label in selected]

    if mode == "all":
        # Still one primitive solve per shape.  "all" means deterministic cycling
        # through all selected labels, not composing labels in one sample.
        return [labels[index % len(labels)] for index in range(n_shapes)]

    if mode == "random_k":
        raise ValueError(
            "label_mode='random_k' is not supported: gear randomizer uses exactly "
            "one primitive slippage label per output shape"
        )

    raise ValueError("label_mode must be: balanced, random_one, or all")


def make_jobs(
    meta: Dict[str, Any],
    *,
    labels: Optional[Sequence[str]] = None,
    num_shapes: Optional[int] = None,
    seed: Optional[int] = None,
    label_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sampler_cfg = meta.get("sampler", {})
    n_shapes = int(
        num_shapes if num_shapes is not None else sampler_cfg.get("n_shapes", 1)
    )
    base_seed = int(seed if seed is not None else sampler_cfg.get("seed", 0))
    mode = str(
        label_mode if label_mode is not None else sampler_cfg.get("label_mode", "balanced")
    )
    sampling_mode = str(sampler_cfg.get("value_sampling", "linspace")).lower()

    specs = parse_deformations(meta)
    selected_labels = list(labels) if labels else sampler_cfg.get("labels")
    schedule_rng = np.random.default_rng(base_seed)
    schedule = choose_labels(specs, selected_labels, n_shapes, mode, schedule_rng)

    label_totals = Counter(schedule)
    label_seen: Dict[str, int] = defaultdict(int)

    jobs: List[Dict[str, Any]] = []
    for sample_id, label in enumerate(schedule):
        sample_seed = int(base_seed + 1_000_003 * sample_id)
        sample_rng = np.random.default_rng(sample_seed)
        linspace_index = label_seen[label]
        linspace_count = label_totals[label]
        label_seen[label] += 1

        operation = sample_deformation(
            specs[label],
            rng=sample_rng,
            sampling_mode=sampling_mode,
            linspace_index=linspace_index,
            linspace_count=linspace_count,
        )
        jobs.append(
            {
                "sample_id": sample_id,
                "seed": sample_seed,
                "label": label,
                "operation": operation,
                "meta": meta,
            }
        )
    return jobs


# -----------------------------------------------------------------------------
# Output and grasp helpers
# -----------------------------------------------------------------------------


def output_paths(meta: Dict[str, Any], sample_name: str) -> Dict[str, Path]:
    output_cfg = meta.get("output", {})
    root = Path(output_cfg.get("root", "Gear_aug_outputs"))
    layout = str(output_cfg.get("layout", "per_shape_dir"))
    if layout not in {"per_shape_dir", "flat"}:
        raise ValueError("output.layout must be 'per_shape_dir' or 'flat'")
    sample_dir = root / sample_name if layout == "per_shape_dir" else root

    return {
        "root": root,
        "sample_dir": sample_dir,
        "final_obj": sample_dir / f"{sample_name}.obj",
        "raw_obj": sample_dir / f"{sample_name}_raw_deformed.obj",
        "grasp": sample_dir / f"{sample_name}_grasp.yaml",
        "sample_meta": sample_dir / f"{sample_name}_sample.yaml",
        "debug": sample_dir / f"{sample_name}_debug.yaml",
        "hole_spec": sample_dir / f"{sample_name}_hole_constraint.json",
    }


def _load_optional_mapping(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        data = load_meta(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_grasp_yaml(
    path: str | Path,
    *,
    grasp_result: Dict[str, Any],
    mesh_path: str | Path,
    source_grasp_path: str | Path,
    operation: Dict[str, Any],
) -> Path:
    """Write transferred grasp data in the format used by the existing loader."""
    if not isinstance(grasp_result, dict):
        raise TypeError(f"grasp_result must be a dict, got {type(grasp_result).__name__}")

    source = _load_optional_mapping(str(source_grasp_path))
    T_mesh_hand_tcp = grasp_result.get("T_mesh_hand_tcp")
    T_mesh_hand = grasp_result.get("T_mesh_hand", T_mesh_hand_tcp)
    if T_mesh_hand_tcp is None and T_mesh_hand is None:
        raise KeyError("grasp_result is missing T_mesh_hand_tcp / T_mesh_hand")
    if T_mesh_hand_tcp is None:
        T_mesh_hand_tcp = T_mesh_hand
    if T_mesh_hand is None:
        T_mesh_hand = T_mesh_hand_tcp

    opening = float(
        source.get(
            "opening_width_m",
            source.get(
                "opening",
                grasp_result.get("opening_width_m", grasp_result.get("opening", 0.06)),
            ),
        )
    )
    record = {
        "mesh_path": str(mesh_path),
        "reference_frame": str(source.get("reference_frame", "mesh_local_frame")),
        "hand_frame": str(source.get("hand_frame", "panda_hand")),
        "opening_width_m": opening,
        "pregrasp_opening_width_m": float(
            source.get("pregrasp_opening_width_m", source.get("pregrasp_opening", opening))
        ),
        "finger_joint_m": float(
            source.get("finger_joint_m", source.get("finger_joint", opening / 2.0))
        ),
        "T_mesh_hand": matrix4(T_mesh_hand, "T_mesh_hand"),
        "T_mesh_hand_tcp": matrix4(T_mesh_hand_tcp, "T_mesh_hand_tcp"),
        "source_grasp_path": str(source_grasp_path),
        "deformation": {
            "label": operation["label"],
            "method": "slippage",
            "magnitudes": operation["magnitudes"],
        },
    }
    return dump_yaml_or_json(path, record)


def method_max_iters(spec: DeformationSpec, solver_cfg: Dict[str, Any]) -> int:
    if spec.max_iters is not None:
        return int(spec.max_iters)

    mapping = solver_cfg.get("method_max_iters", {})
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if str(key).lower() == "slippage":
                return int(value)

    if "slippage_max_iters" in solver_cfg:
        return int(solver_cfg["slippage_max_iters"])
    return int(solver_cfg.get("max_iters", 50))


# -----------------------------------------------------------------------------
# Worker and runner
# -----------------------------------------------------------------------------


def _worker_generate_shape(job: Dict[str, Any]) -> Dict[str, Any]:
    """Picklable worker: fresh FPSA and hole-constraint objects per sample."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    sample_id = int(job.get("sample_id", -1))
    try:
        # Import only in child processes to avoid sharing C++/Open3D/libigl state.
        from FPSA import ShapeAugmentor
        from gear_hole_constraint import GearHoleHardConstraint

        meta = job["meta"]
        operation = job["operation"]
        specs = parse_deformations(meta)
        spec = specs[operation["label"]]
        object_cfg = meta.get("object", {})
        solver_cfg = meta.get("solver", {})
        output_cfg = meta.get("output", {})

        obj_path = resolve_obj_path(meta)
        initial_grasp_path = resolve_initial_grasp_path(meta)
        if not Path(obj_path).is_file():
            raise FileNotFoundError(f"Input OBJ not found: {obj_path}")
        if initial_grasp_path is not None and not Path(initial_grasp_path).is_file():
            raise FileNotFoundError(f"Initial grasp YAML not found: {initial_grasp_path}")

        object_name = safe_name(str(object_cfg.get("name", Path(obj_path).stem)))
        label_tag = safe_name(operation["label"])
        sample_name = f"{object_name}_{label_tag}_{sample_id:06d}"
        paths = output_paths(meta, sample_name)
        paths["sample_dir"].mkdir(parents=True, exist_ok=True)

        protected = [paths["final_obj"], paths["sample_meta"]]
        if initial_grasp_path is not None and bool(output_cfg.get("save_grasp", True)):
            protected.append(paths["grasp"])
        existing = [path for path in protected if path.exists()]
        if existing and not bool(output_cfg.get("overwrite", False)):
            raise FileExistsError(f"Outputs exist and overwrite=false: {existing}")

        augmentor = ShapeAugmentor(
            obj_path=obj_path,
            initial_grasp_path=initial_grasp_path,
        )

        hole_cfg = resolve_hole_constraint_config(meta)
        post_enforce = bool(hole_cfg.pop("post_enforce"))
        hole_constraint = GearHoleHardConstraint.from_augmentor(
            augmentor,
            **hole_cfg,
        )

        topology_report = hole_constraint.topology_report(augmentor.F)
        target_report = hole_constraint.target_displacement_report(augmentor.V)
        max_iters = method_max_iters(spec, solver_cfg)
        handle_error = (
            spec.handle_error_distrib_enabled
            if spec.handle_error_distrib_enabled is not None
            else bool(solver_cfg.get("handle_error_distrib_enabled", False))
        )

        # Exactly one constrained slippage call per output shape.
        V_solver = hole_constraint.displacement_reshape(
            augmentor=augmentor,
            constraint_ids=operation["constraint_ids"],
            displace_idxs=operation["reshaped_ids"],
            displacements=np.asarray(operation["displacements"], dtype=np.float64),
            max_iters=max_iters,
            handle_error_distrib_enabled=bool(handle_error),
            input_name=sample_name,
            reshape_method="slippage",
            post_enforce=False,
        )

        raw_obj_path: Optional[str] = None
        if bool(output_cfg.get("save_raw_deformed", False)):
            augmentor.write_augment_obj(
                output_path=str(paths["raw_obj"]),
                write_coacd=False,
            )
            raw_obj_path = str(paths["raw_obj"])

        if post_enforce:
            V_final = hole_constraint.enforce_on_augmentor(
                augmentor,
                vertices=V_solver,
            )
        else:
            V_final = np.asarray(V_solver, dtype=np.float64)

        diameter_report = hole_constraint.diameter_report(V_final)

        grasp_result: Optional[Dict[str, Any]] = None
        anchor: Any = None
        transfer_debug: Any = None
        grasp_path = ""
        if initial_grasp_path is not None and bool(output_cfg.get("save_grasp", True)):
            grasp_result, anchor, transfer_debug = augmentor.transfer_initial_grasp_guess(
                k_ring=int(solver_cfg.get("k_ring", 3)),
                use_distance_weights=bool(
                    solver_cfg.get("use_distance_weights", True)
                ),
                quat_order=str(solver_cfg.get("quat_order", "xyzw")),
                patch_method=str(solver_cfg.get("patch_method", "k_ring")),
                return_format="dict",
            )

            if bool(solver_cfg.get("visualize_grasp_transfer", False)):
                augmentor.visualize_deformed_grasp_pose(
                    T_grasp_new=grasp_result["T_mesh_hand_tcp"],
                    anchor=anchor,
                    debug_info=transfer_debug,
                    show_anchor=True,
                    show_patch=True,
                    show_old_grasp=True,
                )

        final_obj_path, coacd_path = augmentor.write_augment_obj(
            output_path=str(paths["final_obj"]),
            write_coacd=bool(output_cfg.get("write_coacd", True)),
            return_paths=True,
        )

        if grasp_result is not None and initial_grasp_path is not None:
            grasp_path = str(
                write_grasp_yaml(
                    paths["grasp"],
                    grasp_result=grasp_result,
                    mesh_path=final_obj_path,
                    source_grasp_path=initial_grasp_path,
                    operation=operation,
                )
            )

        hole_constraint.save_spec(paths["hole_spec"])
        sample_record = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "seed": int(job["seed"]),
            "label": operation["label"],
            "method": "slippage",
            "sampling_strategy": "paired",
            "operation": operation,
            "used_max_iters": max_iters,
            "handle_error_distrib_enabled": bool(handle_error),
            "source_obj_path": obj_path,
            "source_grasp_path": initial_grasp_path,
            "final_obj_path": final_obj_path,
            "coacd_path": coacd_path,
            "raw_obj_path": raw_obj_path,
            "grasp_path": grasp_path or None,
            "hole_constraint_path": str(paths["hole_spec"]),
            "post_enforce": post_enforce,
            "hole_topology_report": topology_report,
            "hole_target_report": target_report,
            "hole_diameter_report": diameter_report,
            "transferred_grasp": grasp_result,
        }
        sample_meta_path = dump_yaml_or_json(paths["sample_meta"], sample_record)

        debug_path = ""
        if bool(output_cfg.get("save_debug", True)):
            debug_record = {
                "sample": sample_record,
                "mesh_status": (
                    augmentor.mesh_status() if hasattr(augmentor, "mesh_status") else None
                ),
                "grasp_anchor": anchor,
                "grasp_transfer_debug": transfer_debug,
                "hole_vertex_ids": hole_constraint.vertex_ids,
                "hole_center": hole_constraint.center,
                "hole_axis": hole_constraint.axis,
                "hole_radius": hole_constraint.radius,
            }
            debug_path = str(dump_yaml_or_json(paths["debug"], debug_record))

        return {
            "ok": True,
            "sample_id": sample_id,
            "sample_name": sample_name,
            "label": operation["label"],
            "obj_path": final_obj_path,
            "coacd_path": coacd_path or "",
            "grasp_path": grasp_path,
            "meta_path": str(sample_meta_path),
            "debug_path": debug_path,
            "max_hole_radius_drift": diameter_report["max_radius_drift"],
            "error": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "sample_id": sample_id,
            "sample_name": "",
            "label": str(job.get("label", "")),
            "obj_path": "",
            "coacd_path": "",
            "grasp_path": "",
            "meta_path": "",
            "debug_path": "",
            "max_hole_radius_drift": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def write_manifest(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "manifest.csv"
    jsonl_path = output_root / "manifest.jsonl"
    fieldnames = [
        "ok",
        "sample_id",
        "sample_name",
        "label",
        "obj_path",
        "coacd_path",
        "grasp_path",
        "meta_path",
        "debug_path",
        "max_hole_radius_drift",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def apply_cli_overrides(
    meta: Dict[str, Any],
    *,
    obj_path: Optional[str],
    initial_grasp_path: Optional[str],
    output_root: Optional[str],
) -> Dict[str, Any]:
    result = copy.deepcopy(meta)
    if obj_path is not None:
        result.setdefault("object", {})["obj_path"] = obj_path
    if initial_grasp_path is not None:
        result.setdefault("object", {})["initial_grasp_path"] = initial_grasp_path
    if output_root is not None:
        result.setdefault("output", {})["root"] = output_root
    return result


def run_batch(
    meta_path: str | Path,
    *,
    labels: Optional[Sequence[str]] = None,
    num_shapes: Optional[int] = None,
    workers: Optional[int] = None,
    seed: Optional[int] = None,
    label_mode: Optional[str] = None,
    obj_path: Optional[str] = None,
    initial_grasp_path: Optional[str] = None,
    output_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    meta = apply_cli_overrides(
        load_meta(meta_path),
        obj_path=obj_path,
        initial_grasp_path=initial_grasp_path,
        output_root=output_root,
    )
    # Validate constraint settings before starting child processes.
    resolve_hole_constraint_config(meta)
    jobs = make_jobs(
        meta,
        labels=labels,
        num_shapes=num_shapes,
        seed=seed,
        label_mode=label_mode,
    )
    if not jobs:
        return []

    output_root_path = Path(meta.get("output", {}).get("root", "Gear_aug_outputs"))
    output_root_path.mkdir(parents=True, exist_ok=True)

    sampler_cfg = meta.get("sampler", {})
    requested_workers = int(
        workers
        if workers is not None
        else sampler_cfg.get("max_workers", max(1, os.cpu_count() or 1))
    )
    max_workers = max(1, min(requested_workers, len(jobs)))
    start_method = str(sampler_cfg.get("mp_start_method", "spawn"))
    context = mp.get_context(start_method)

    rows: List[Dict[str, Any]] = []
    if max_workers == 1:
        rows = [_worker_generate_shape(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(_worker_generate_shape, job) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: int(row.get("sample_id", 10**12)))
    write_manifest(output_root_path, rows)

    succeeded = sum(bool(row.get("ok")) for row in rows)
    print(
        f"[FPSA gear batch] done: {succeeded}/{len(rows)} succeeded; "
        f"manifest: {output_root_path / 'manifest.csv'}"
    )
    if succeeded != len(rows):
        print(
            f"[FPSA gear batch] failed: {len(rows) - succeeded}; "
            f"see {output_root_path / 'manifest.jsonl'}"
        )
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def split_labels(value: Optional[str]) -> Optional[List[str]]:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def dry_run_summary(jobs: Sequence[Dict[str, Any]], limit: int = 8) -> Dict[str, Any]:
    label_counts = Counter(str(job["label"]) for job in jobs)
    preview = []
    for job in jobs[:limit]:
        operation = job["operation"]
        preview.append(
            {
                "sample_id": job["sample_id"],
                "seed": job["seed"],
                "label": job["label"],
                "sampling_strategy": operation["sampling_strategy"],
                "linspace_index": operation["linspace_index"],
                "linspace_count": operation["linspace_count"],
                "magnitudes": operation["magnitudes"],
                "displacements": operation["displacements"],
            }
        )
    return {
        "num_jobs": len(jobs),
        "jobs_per_label": dict(label_counts),
        "note": "One paired handle sample and one constrained slippage solve per job; no chain and no Cartesian handle product.",
        "preview": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-stage constrained FPSA randomizer for gear deformation"
    )
    parser.add_argument(
        "--meta",
        default="gear_streching.yaml",
        help="Path to gear_streching YAML/JSON meta file",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated primitive labels; overrides sampler.labels",
    )
    parser.add_argument("--num-shapes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--label-mode",
        choices=["balanced", "random_one", "all"],
        default=None,
    )
    parser.add_argument("--obj-path", default=None, help="Override object.obj_path")
    parser.add_argument(
        "--initial-grasp-path",
        default=None,
        help="Override object.initial_grasp_path",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output.root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview sampling without importing FPSA",
    )
    args = parser.parse_args()

    labels = split_labels(args.labels)
    if args.dry_run:
        meta = apply_cli_overrides(
            load_meta(args.meta),
            obj_path=args.obj_path,
            initial_grasp_path=args.initial_grasp_path,
            output_root=args.output_root,
        )
        hole_cfg = resolve_hole_constraint_config(meta)
        jobs = make_jobs(
            meta,
            labels=labels,
            num_shapes=args.num_shapes,
            seed=args.seed,
            label_mode=args.label_mode,
        )
        summary = dry_run_summary(jobs)
        summary["hole_constraint"] = hole_cfg
        sampler_cfg = meta.get("sampler", {})
        if str(sampler_cfg.get("sampling_strategy", "paired")).lower() == "cartesian":
            summary["sampling_strategy_warning"] = (
                "sampler.sampling_strategy=cartesian is intentionally ignored; "
                "this randomizer always uses paired handle sampling."
            )
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
        return

    run_batch(
        meta_path=args.meta,
        labels=labels,
        num_shapes=args.num_shapes,
        workers=args.workers,
        seed=args.seed,
        label_mode=args.label_mode,
        obj_path=args.obj_path,
        initial_grasp_path=args.initial_grasp_path,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
