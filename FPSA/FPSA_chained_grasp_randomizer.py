"""
Multiprocessing batch randomizer for chained FPSA assembly-tool deformation.

For every sampled deformation this runner:
  1. Creates a fresh ``FPSA.ShapeAugmentor`` with the source mesh and its
     initial-grasp YAML.
  2. Applies the selected primitive stages sequentially.
  3. Transfers the original grasp initial guess to the final deformed mesh.
  4. Saves the augmented visual mesh, cached convex-decomposed mesh, and the
     transferred grasp initial guess YAML.

Cartesian linspace sampling is supported.  For a two-stage chain such as
``chain_x_and_slippage``, ``linspace_points_per_stage: N`` produces the full
independent N x N grid (every stage-1 value paired with every stage-2 value).

Example:
    python FPSA_chained_grasp_randomizer.py \
        --meta assembly_streching.yaml \
        --labels chain_x_and_slippage \
        --workers 8 \
        --output-root assemblyTool_aug_outputs
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
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_OBJ_PATH = "/home/iadc/GeoBridge/data/objects/assembly/tool/tool.obj"
DEFAULT_INITIAL_GRASP_PATH = "/home/iadc/GeoBridge/data/objects/assembly/tool/tool_grasp.yaml"
DEFAULT_METHOD_MAX_ITERS = {"apap": 200, "slippage": 120}


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
    """Convert common scientific-Python values to YAML/JSON-safe values."""
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
    # Debug objects from geometry libraries are not always directly serializable.
    return repr(value)


def dump_yaml_or_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_builtin(data)

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            path = path.with_suffix(".json")
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def unique_ints(values: Iterable[int]) -> List[int]:
    seen: set[int] = set()
    result: List[int] = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_.")
    return cleaned or "sample"


def matrix4(value: Any, name: str) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got shape {mat.shape}")
    return mat


# -----------------------------------------------------------------------------
# Meta model: primitive deformation stages and chain labels
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveSpec:
    label: str
    method: str
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
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainSpec:
    label: str
    steps: List[str]
    type: str = "chain"
    weight: float = 1.0
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


Spec = Union[PrimitiveSpec, ChainSpec]


def parse_chain_steps(value: Any, label: str) -> List[str]:
    if isinstance(value, str):
        steps = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        steps = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError(f"Chain '{label}' must use a comma-separated string or list")

    if not steps:
        raise ValueError(f"Chain '{label}' has no stages")
    return steps


def parse_specs(meta: Dict[str, Any]) -> Dict[str, Spec]:
    raw_deformations = meta.get("deformations")
    if not isinstance(raw_deformations, list) or not raw_deformations:
        raise ValueError("meta['deformations'] must be a non-empty list")

    specs: Dict[str, Spec] = {}
    for raw in raw_deformations:
        if not isinstance(raw, dict):
            raise TypeError("Every deformation entry must be a mapping")
        if "label" not in raw:
            raise KeyError("Every deformation entry requires 'label'")

        label = str(raw["label"])
        if label in specs:
            raise ValueError(f"Duplicated deformation label: {label}")

        typ = str(raw.get("type", "displacement")).strip()
        if typ.lower() == "chain":
            known = {"label", "type", "chain", "weight", "description"}
            specs[label] = ChainSpec(
                label=label,
                steps=parse_chain_steps(raw.get("chain"), label),
                weight=float(raw.get("weight", 1.0)),
                description=str(raw.get("description", "")),
                extra={k: v for k, v in raw.items() if k not in known},
            )
            continue

        required = ["constrained_ids", "reshaped_ids", "reshaped_vector", "range"]
        missing = [key for key in required if key not in raw]
        if missing:
            raise KeyError(f"Primitive deformation '{label}' is missing: {missing}")

        method = str(raw.get("method", typ if typ.lower() != "displacement" else "slippage"))
        known = {
            "label", "type", "method", "constrained_ids", "reshaped_ids",
            "reshaped_vector", "range", "coupled", "normalize_vector",
            "distribution", "weight", "max_iters", "description",
        }
        specs[label] = PrimitiveSpec(
            label=label,
            method=method,
            constrained_ids=[int(v) for v in raw["constrained_ids"]],
            reshaped_ids=[int(v) for v in raw["reshaped_ids"]],
            reshaped_vector=raw["reshaped_vector"],
            range=raw["range"],
            type=typ,
            coupled=bool(raw.get("coupled", True)),
            normalize_vector=bool(raw.get("normalize_vector", True)),
            distribution=str(raw.get("distribution", "uniform")),
            weight=float(raw.get("weight", 1.0)),
            max_iters=int(raw["max_iters"]) if raw.get("max_iters") is not None else None,
            description=str(raw.get("description", "")),
            extra={k: v for k, v in raw.items() if k not in known},
        )

    # Validate all references and detect cycles up front.
    for label in specs:
        resolve_primitive_labels(label, specs)
    return specs


def resolve_primitive_labels(
    label: str,
    specs: Dict[str, Spec],
    stack: Optional[List[str]] = None,
) -> List[str]:
    """Recursively flatten a chain while preserving the declared stage order."""
    if label not in specs:
        raise KeyError(f"Unknown deformation label '{label}'. Available: {list(specs)}")

    stack = list(stack or [])
    if label in stack:
        cycle = " -> ".join([*stack, label])
        raise ValueError(f"Cyclic chain definition detected: {cycle}")

    spec = specs[label]
    if isinstance(spec, PrimitiveSpec):
        return [label]

    result: List[str] = []
    next_stack = [*stack, label]
    for child in spec.steps:
        result.extend(resolve_primitive_labels(child, specs, next_stack))
    return result


def default_selectable_labels(specs: Dict[str, Spec]) -> List[str]:
    """Prefer chain labels; fall back to primitive labels when no chain exists."""
    chains = [label for label, spec in specs.items() if isinstance(spec, ChainSpec)]
    return chains if chains else list(specs)


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


def vector_for_index(vector_spec: Any, index: int, count: int, normalize: bool) -> np.ndarray:
    axis = {
        "x": [1.0, 0.0, 0.0], "+x": [1.0, 0.0, 0.0], "-x": [-1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0], "+y": [0.0, 1.0, 0.0], "-y": [0.0, -1.0, 0.0],
        "z": [0.0, 0.0, 1.0], "+z": [0.0, 0.0, 1.0], "-z": [0.0, 0.0, -1.0],
    }
    if isinstance(vector_spec, str):
        key = vector_spec.lower().strip()
        if key not in axis:
            raise ValueError(f"Unknown reshaped_vector axis: {vector_spec}")
        return normalize_vec(axis[key], normalize)

    arr = np.asarray(vector_spec, dtype=np.float64)
    if arr.shape == (3,):
        return normalize_vec(arr, normalize)
    if arr.shape == (1, 3):  # broadcast one vector to every reshaped vertex
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
        # One common numeric [low, high] range.
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            return values
        # Broadcast one nested range, as used by wrench_quality_stretch.yaml.
        if len(values) == 1:
            return values[0]
        # One range per reshaped vertex.
        if len(values) == count:
            return values[index]

    arr = np.asarray(range_spec, dtype=object)
    if arr.shape == (2,):
        return range_spec
    raise ValueError(
        f"range must be [low, high], one broadcast range, or {count} per-id ranges; got {range_spec}"
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
    sampling_mode = sampling_mode.lower()
    if linspace_count <= 0:
        raise ValueError("linspace_count must be positive")

    def linear_value(low: float, high: float) -> float:
        low, high = float(low), float(high)
        if sampling_mode == "random":
            return float(rng.uniform(low, high))
        if sampling_mode != "linspace":
            raise ValueError("sampler.value_sampling must be 'linspace' or 'random'")
        if linspace_count == 1:
            return float(0.5 * (low + high))
        return float(np.linspace(low, high, linspace_count, dtype=np.float64)[linspace_index])

    if isinstance(range_spec, dict):
        typ = str(range_spec.get("type", distribution)).lower()
        if typ == "uniform":
            return linear_value(range_spec["low"], range_spec["high"])
        if typ == "fixed":
            return float(range_spec["value"])
        if typ == "choice":
            choices = list(range_spec.get("values", []))
            if not choices:
                raise ValueError("choice range requires non-empty 'values'")
            if sampling_mode == "random":
                return float(rng.choice(np.asarray(choices, dtype=np.float64)))
            return float(choices[linspace_index % len(choices)])
        if typ == "normal":
            if sampling_mode != "random":
                raise ValueError("normal distribution requires sampler.value_sampling: random")
            mean = float(range_spec.get("mean", 0.0))
            std = float(range_spec["std"])
            value = float(rng.normal(mean, std))
            if "low" in range_spec:
                value = max(value, float(range_spec["low"]))
            if "high" in range_spec:
                value = min(value, float(range_spec["high"]))
            return value
        raise ValueError(f"Unsupported range type: {typ}")

    arr = np.asarray(range_spec, dtype=np.float64)
    if arr.shape == (2,):
        return linear_value(float(arr[0]), float(arr[1]))
    raise ValueError(f"Invalid scalar range: {range_spec}")


def sample_primitive(
    spec: PrimitiveSpec,
    *,
    sampling_mode: str,
    linspace_index: int,
    linspace_count: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n_handles = len(spec.reshaped_ids)
    if n_handles == 0:
        raise ValueError(f"Primitive '{spec.label}' has no reshaped_ids")

    if spec.coupled:
        common_mag = scalar_from_range(
            range_for_index(spec.range, 0, n_handles),
            sampling_mode=sampling_mode,
            linspace_index=linspace_index,
            linspace_count=linspace_count,
            rng=rng,
            distribution=spec.distribution,
        )
        magnitudes = [common_mag] * n_handles
    else:
        magnitudes = [
            scalar_from_range(
                range_for_index(spec.range, i, n_handles),
                sampling_mode=sampling_mode,
                linspace_index=linspace_index,
                linspace_count=linspace_count,
                rng=rng,
                distribution=spec.distribution,
            )
            for i in range(n_handles)
        ]

    displacements: List[List[float]] = []
    for i, magnitude in enumerate(magnitudes):
        direction = vector_for_index(
            spec.reshaped_vector,
            i,
            n_handles,
            spec.normalize_vector,
        )
        displacements.append((float(magnitude) * direction).tolist())

    # FPSA requires displace_idxs to be contained in constraint_ids.
    constraint_ids = unique_ints([*spec.constrained_ids, *spec.reshaped_ids])
    return {
        "label": spec.label,
        "method": spec.method,
        "constraint_ids": constraint_ids,
        "reshaped_ids": list(spec.reshaped_ids),
        "displacements": displacements,
        "magnitudes": [float(v) for v in magnitudes],
        "max_iters": spec.max_iters,
        "description": spec.description,
    }


def choose_top_labels(
    specs: Dict[str, Spec],
    selected_labels: Optional[Sequence[str]],
    n_shapes: int,
    mode: str,
    rng: np.random.Generator,
    labels_per_shape: int,
) -> List[List[str]]:
    labels = list(selected_labels) if selected_labels else default_selectable_labels(specs)
    unknown = [label for label in labels if label not in specs]
    if unknown:
        raise KeyError(f"Unknown labels: {unknown}. Available: {list(specs)}")
    if not labels or n_shapes <= 0:
        return []

    mode = mode.lower()
    labels_per_shape = max(1, int(labels_per_shape))

    if mode == "balanced":
        schedule = [[labels[i % len(labels)]] for i in range(n_shapes)]
        rng.shuffle(schedule)
        return schedule

    if mode == "random_one":
        weights = np.asarray([max(0.0, float(specs[label].weight)) for label in labels])
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(labels), dtype=np.float64)
        probabilities = weights / weights.sum()
        indices = rng.choice(len(labels), size=n_shapes, replace=True, p=probabilities)
        return [[labels[int(i)]] for i in indices]

    if mode == "random_k":
        k = min(labels_per_shape, len(labels))
        return [list(rng.choice(labels, size=k, replace=False)) for _ in range(n_shapes)]

    if mode == "all":
        return [list(labels) for _ in range(n_shapes)]

    raise ValueError("label_mode must be balanced, random_one, random_k, or all")


def _points_for_primitive(
    sampler_cfg: Dict[str, Any],
    primitive_label: str,
) -> int:
    """Resolve the number of linspace points for one primitive stage.

    ``sampler.linspace_points_per_stage`` may be either one integer or a mapping:

        linspace_points_per_stage: 8

    or:

        linspace_points_per_stage:
          default: 8
          APAP_y_stretch: 10
          slippage_x_stretch: 12
    """
    raw = sampler_cfg.get("linspace_points_per_stage", 8)
    if isinstance(raw, dict):
        value = raw.get(primitive_label, raw.get("default", 8))
    else:
        value = raw
    points = int(value)
    if points <= 0:
        raise ValueError(
            f"linspace_points_per_stage for '{primitive_label}' must be positive, got {points}"
        )
    return points


def _selected_top_labels(
    specs: Dict[str, Spec],
    labels: Optional[Sequence[str]],
    sampler_cfg: Dict[str, Any],
) -> List[str]:
    selected = list(labels) if labels else sampler_cfg.get("labels")
    result = list(selected) if selected else default_selectable_labels(specs)
    unknown = [label for label in result if label not in specs]
    if unknown:
        raise KeyError(f"Unknown labels: {unknown}. Available: {list(specs)}")
    return result


def _make_cartesian_jobs(
    meta: Dict[str, Any],
    specs: Dict[str, Spec],
    *,
    labels: Optional[Sequence[str]],
    num_shapes: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    """Generate a complete independent linspace grid for every selected chain.

    For a two-stage chain with eight points per stage, this produces all 64 pairs:

        (APAP index 0, slippage index 0..7)
        (APAP index 1, slippage index 0..7)
        ...
        (APAP index 7, slippage index 0..7)

    Therefore every APAP deformation receives the complete slippage sweep.

    In Cartesian mode, sampler.n_shapes / --num-shapes denotes the TOTAL across
    all selected top-level labels. Mixed chain lengths are supported: four
    two-stage 5x5 chains plus one single-stage 5-point primitive produce 105.
    """
    sampler_cfg = meta.get("sampler", {})
    sampling_mode = str(sampler_cfg.get("value_sampling", "linspace")).lower()
    if sampling_mode != "linspace":
        raise ValueError(
            "sampler.sampling_strategy='cartesian' requires value_sampling='linspace'"
        )

    base_seed = int(seed if seed is not None else sampler_cfg.get("seed", 0))
    selected = _selected_top_labels(specs, labels, sampler_cfg)

    configured_expected = num_shapes
    if configured_expected is None and sampler_cfg.get("n_shapes") is not None:
        configured_expected = int(sampler_cfg["n_shapes"])

    # Resolve every selected top-level label first. A top label may expand to
    # a multi-stage chain (for example 5 x 5 = 25 samples) or to one primitive
    # (for example 5 samples). ``n_shapes`` is validated against the TOTAL batch
    # size so both kinds can be selected together.
    label_grids: List[Tuple[str, List[str], List[int], int]] = []
    for top_label in selected:
        primitive_labels = resolve_primitive_labels(top_label, specs)
        stage_counts = [
            _points_for_primitive(sampler_cfg, primitive_label)
            for primitive_label in primitive_labels
        ]
        combinations = int(np.prod(stage_counts, dtype=np.int64))
        label_grids.append((top_label, primitive_labels, stage_counts, combinations))

    total_combinations = sum(item[3] for item in label_grids)
    if configured_expected is not None and int(configured_expected) != total_combinations:
        breakdown = ", ".join(
            f"{label}={count}" for label, _stages, _counts, count in label_grids
        )
        raise ValueError(
            f"Cartesian selection generates {total_combinations} total samples "
            f"({breakdown}), but n_shapes/--num-shapes is {configured_expected}. "
            f"Set it to {total_combinations}, or remove it."
        )

    jobs: List[Dict[str, Any]] = []
    sample_id = 0
    for top_label, primitive_labels, stage_counts, combinations_per_chain in label_grids:
        index_ranges = [range(count) for count in stage_counts]
        for grid_indices_tuple in product(*index_ranges):
            sample_seed = int(base_seed + 1_000_003 * sample_id)
            sample_rng = np.random.default_rng(sample_seed)
            sampled_stages: List[Dict[str, Any]] = []

            for stage_position, (primitive_label, index, count) in enumerate(
                zip(primitive_labels, grid_indices_tuple, stage_counts),
                start=1,
            ):
                spec = specs[primitive_label]
                if not isinstance(spec, PrimitiveSpec):
                    raise AssertionError("Resolved chain stage must be primitive")

                stage = sample_primitive(
                    spec,
                    sampling_mode="linspace",
                    linspace_index=int(index),
                    linspace_count=int(count),
                    rng=sample_rng,
                )
                stage.update(
                    {
                        "top_label": top_label,
                        "sampling_strategy": "cartesian",
                        "linspace_scope": "stage_independent",
                        "stage_position": int(stage_position),
                        "linspace_index": int(index),
                        "linspace_count": int(count),
                        "grid_indices": [int(v) for v in grid_indices_tuple],
                        "grid_counts": [int(v) for v in stage_counts],
                    }
                )
                sampled_stages.append(stage)

            jobs.append(
                {
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "top_labels": [top_label],
                    "sampled_stages": sampled_stages,
                    "sampling_strategy": "cartesian",
                    "grid_indices": [int(v) for v in grid_indices_tuple],
                    "grid_counts": [int(v) for v in stage_counts],
                    "combinations_per_chain": combinations_per_chain,
                    "meta": meta,
                }
            )
            sample_id += 1

    return jobs


def make_jobs(
    meta: Dict[str, Any],
    *,
    labels: Optional[Sequence[str]] = None,
    num_shapes: Optional[int] = None,
    seed: Optional[int] = None,
    label_mode: Optional[str] = None,
    labels_per_shape: Optional[int] = None,
) -> List[Dict[str, Any]]:
    specs = parse_specs(meta)
    sampler_cfg = meta.get("sampler", {})
    sampling_strategy = str(sampler_cfg.get("sampling_strategy", "paired")).lower()

    if sampling_strategy == "cartesian":
        return _make_cartesian_jobs(
            meta,
            specs,
            labels=labels,
            num_shapes=num_shapes,
            seed=seed,
        )
    if sampling_strategy not in {"paired", "synchronized"}:
        raise ValueError(
            "sampler.sampling_strategy must be 'paired'/'synchronized' or 'cartesian'"
        )

    # Legacy paired behavior: one output job contains one value from every stage.
    n_shapes = int(num_shapes if num_shapes is not None else sampler_cfg.get("n_shapes", 1))
    base_seed = int(seed if seed is not None else sampler_cfg.get("seed", 0))
    mode = str(label_mode if label_mode is not None else sampler_cfg.get("label_mode", "balanced"))
    k = int(
        labels_per_shape
        if labels_per_shape is not None
        else sampler_cfg.get("labels_per_shape", 1)
    )
    sampling_mode = str(sampler_cfg.get("value_sampling", "linspace")).lower()

    selected = list(labels) if labels else sampler_cfg.get("labels")
    schedule_rng = np.random.default_rng(base_seed)
    top_schedule = choose_top_labels(specs, selected, n_shapes, mode, schedule_rng, k)

    linspace_scope = str(sampler_cfg.get("linspace_scope", "chain")).lower()
    if linspace_scope not in {"chain", "primitive_global"}:
        raise ValueError("sampler.linspace_scope must be 'chain' or 'primitive_global'")

    top_total = Counter(label for top_labels in top_schedule for label in top_labels)
    top_seen: defaultdict[str, int] = defaultdict(int)

    expanded_schedule: List[List[str]] = []
    for top_labels in top_schedule:
        stages: List[str] = []
        for top_label in top_labels:
            stages.extend(resolve_primitive_labels(top_label, specs))
        expanded_schedule.append(stages)

    primitive_total = Counter(label for stages in expanded_schedule for label in stages)
    primitive_seen: defaultdict[str, int] = defaultdict(int)

    jobs: List[Dict[str, Any]] = []
    for sample_id, top_labels in enumerate(top_schedule):
        sample_seed = int(base_seed + 1_000_003 * sample_id)
        sample_rng = np.random.default_rng(sample_seed)
        sampled_stages: List[Dict[str, Any]] = []

        for top_label in top_labels:
            chain_index = top_seen[top_label]
            chain_count = top_total[top_label]
            top_seen[top_label] += 1

            for primitive_label in resolve_primitive_labels(top_label, specs):
                spec = specs[primitive_label]
                if not isinstance(spec, PrimitiveSpec):
                    raise AssertionError("Resolved chain stage must be primitive")

                if linspace_scope == "chain":
                    index = chain_index
                    count = chain_count
                else:
                    index = primitive_seen[primitive_label]
                    count = primitive_total[primitive_label]
                    primitive_seen[primitive_label] += 1

                stage = sample_primitive(
                    spec,
                    sampling_mode=sampling_mode,
                    linspace_index=index,
                    linspace_count=count,
                    rng=sample_rng,
                )
                stage.update(
                    {
                        "top_label": top_label,
                        "sampling_strategy": "paired",
                        "linspace_scope": linspace_scope,
                        "linspace_index": int(index),
                        "linspace_count": int(count),
                    }
                )
                sampled_stages.append(stage)

        jobs.append(
            {
                "sample_id": sample_id,
                "seed": sample_seed,
                "top_labels": top_labels,
                "sampled_stages": sampled_stages,
                "sampling_strategy": "paired",
                "meta": meta,
            }
        )
    return jobs


# -----------------------------------------------------------------------------
# Runtime configuration and FPSA worker
# -----------------------------------------------------------------------------


def resolve_obj_path(meta: Dict[str, Any]) -> str:
    object_cfg = meta.get("object", {})
    return str(object_cfg.get("obj_path", meta.get("obj_path", DEFAULT_OBJ_PATH)))


def resolve_initial_grasp_path(meta: Dict[str, Any]) -> str:
    """Resolve the source grasp-initial-guess YAML used by ShapeAugmentor."""
    object_cfg = meta.get("object", {})
    grasp_cfg = meta.get("grasp", {})
    candidate = (
        grasp_cfg.get("initial_grasp_path")
        or grasp_cfg.get("path")
        or object_cfg.get("initial_grasp_path")
        or object_cfg.get("grasp_path")
        or meta.get("initial_grasp_path")
    )
    if candidate is not None:
        return str(candidate)

    obj_path = Path(resolve_obj_path(meta))
    derived = obj_path.with_name(f"{obj_path.stem}_grasp.yaml")
    if derived.exists():
        return str(derived)
    return DEFAULT_INITIAL_GRASP_PATH

def method_max_iters(stage: Dict[str, Any], solver_cfg: Dict[str, Any]) -> int:
    if stage.get("max_iters") is not None:
        return int(stage["max_iters"])

    method = str(stage["method"])
    method_map = solver_cfg.get("method_max_iters", {})
    if isinstance(method_map, dict):
        for key, value in method_map.items():
            if str(key).lower() == method.lower():
                return int(value)

    direct_key = f"{method.lower()}_max_iters"
    if direct_key in solver_cfg:
        return int(solver_cfg[direct_key])
    if "max_iters" in solver_cfg:
        return int(solver_cfg["max_iters"])
    return int(DEFAULT_METHOD_MAX_ITERS.get(method.lower(), 120))


def output_paths(meta: Dict[str, Any], sample_name: str) -> Dict[str, Path]:
    output_cfg = meta.get("output", {})
    root = Path(output_cfg.get("root", "fpsa_chained_outputs"))
    layout = str(output_cfg.get("layout", "per_shape_dir"))
    sample_dir = root / sample_name if layout == "per_shape_dir" else root

    return {
        "root": root,
        "sample_dir": sample_dir,
        "final_obj": sample_dir / f"{sample_name}.obj",
        "raw_obj": sample_dir / f"{sample_name}_raw_deformed.obj",
        "grasp": sample_dir / f"{sample_name}_grasp.yaml",
        "sample_meta": sample_dir / f"{sample_name}_sample.yaml",
        "debug": sample_dir / f"{sample_name}_debug.yaml",
    }


def write_grasp_yaml(
    path: str | Path,
    *,
    grasp_result: Dict[str, Any],
    mesh_path: str | Path,
    source_grasp_path: str | Path,
    top_labels: Sequence[str],
    sampled_stages: Sequence[Dict[str, Any]],
) -> Path:
    """Write a transferred grasp in the format consumed by load_initial_grasp_pose()."""
    if not isinstance(grasp_result, dict):
        raise TypeError(f"grasp_result must be a dict, got {type(grasp_result).__name__}")

    T_mesh_hand_tcp = grasp_result.get("T_mesh_hand_tcp")
    T_mesh_hand = grasp_result.get("T_mesh_hand")
    if T_mesh_hand_tcp is None and T_mesh_hand is None:
        raise KeyError("grasp_result is missing T_mesh_hand_tcp / T_mesh_hand")
    if T_mesh_hand_tcp is None:
        T_mesh_hand_tcp = T_mesh_hand
    if T_mesh_hand is None:
        T_mesh_hand = T_mesh_hand_tcp

    opening = float(grasp_result.get("opening_width_m", grasp_result.get("opening", 0.06)))
    record = {
        "mesh_path": str(mesh_path),
        "reference_frame": str(grasp_result.get("reference_frame", "mesh_local_frame")),
        "hand_frame": str(grasp_result.get("hand_frame", "panda_hand")),
        "opening_width_m": opening,
        "pregrasp_opening_width_m": float(
            grasp_result.get("pregrasp_opening_width_m",
                             grasp_result.get("pregrasp_opening", opening))
        ),
        "finger_joint_m": float(
            grasp_result.get("finger_joint_m",
                             grasp_result.get("finger_joint", opening / 2.0))
        ),
        "T_mesh_hand": matrix4(T_mesh_hand, "T_mesh_hand"),
        "T_mesh_hand_tcp": matrix4(T_mesh_hand_tcp, "T_mesh_hand_tcp"),
        # Reproducibility metadata; existing loaders may safely ignore it.
        "source_grasp_path": str(source_grasp_path),
        "chain_labels": list(top_labels),
        "stages": [
            {
                "label": stage["label"],
                "method": stage["method"],
                "magnitudes": stage["magnitudes"],
            }
            for stage in sampled_stages
        ],
    }
    return dump_yaml_or_json(path, record)

def _worker_generate_shape(job: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level picklable worker. A fresh ShapeAugmentor is created per sample."""
    # Avoid multiplying BLAS/OpenMP thread pools inside every process.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    sample_id = int(job.get("sample_id", -1))
    try:
        # Import inside the child process; do not inherit FPSA/Open3D/libigl state.
        from FPSA import ShapeAugmentor

        meta = job["meta"]
        sampled_stages = job["sampled_stages"]
        top_labels = job["top_labels"]
        object_cfg = meta.get("object", {})
        solver_cfg = meta.get("solver", {})
        output_cfg = meta.get("output", {})

        obj_path = resolve_obj_path(meta)
        object_name = safe_name(str(object_cfg.get("name", Path(obj_path).stem)))
        chain_tag = safe_name("+".join(top_labels))
        sample_name = f"{object_name}_{chain_tag}_{sample_id:06d}"
        paths = output_paths(meta, sample_name)
        paths["sample_dir"].mkdir(parents=True, exist_ok=True)

        overwrite = bool(output_cfg.get("overwrite", False))
        protected = [paths["final_obj"], paths["grasp"], paths["sample_meta"]]
        existing = [path for path in protected if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"Outputs exist and overwrite=false: {existing}")

        initial_grasp_path = resolve_initial_grasp_path(meta)
        if not Path(initial_grasp_path).exists():
            raise FileNotFoundError(f"Initial grasp YAML not found: {initial_grasp_path}")
        augmentor = ShapeAugmentor(
            obj_path=obj_path,
            initial_grasp_path=initial_grasp_path,
        )

        stage_records: List[Dict[str, Any]] = []
        for stage_index, stage in enumerate(sampled_stages, start=1):
            input_name = (
                f"{sample_name}_step{stage_index:02d}_"
                f"{safe_name(stage['method'])}_{safe_name(stage['label'])}"
            )
            displacements = np.asarray(stage["displacements"], dtype=np.float64)
            result_vertices = augmentor.displacement_reshape(
                constraint_ids=[int(v) for v in stage["constraint_ids"]],
                displace_idxs=[int(v) for v in stage["reshaped_ids"]],
                displacements=displacements,
                max_iters=method_max_iters(stage, solver_cfg),
                reshape_method=str(stage["method"]),
                input_name=input_name,
            )
            stage_records.append(
                {
                    **stage,
                    "stage_index": stage_index,
                    "input_name": input_name,
                    "result_shape": list(np.asarray(result_vertices).shape),
                    "used_max_iters": method_max_iters(stage, solver_cfg),
                }
            )

        write_coacd = bool(output_cfg.get("write_coacd", True))
        save_raw = bool(output_cfg.get("save_raw_deformed", False))
        if save_raw:
            augmentor.write_augment_obj(
                output_path=str(paths["raw_obj"]),
                write_coacd=False,
            )

        # Transfer the source assembly grasp to the final chained deformation.
        grasp_result, anchor, transfer_debug = augmentor.transfer_initial_grasp_guess(
            k_ring=int(solver_cfg.get("k_ring", 3)),
            use_distance_weights=bool(solver_cfg.get("use_distance_weights", True)),
            quat_order=str(solver_cfg.get("quat_order", "xyzw")),
            patch_method=str(solver_cfg.get("patch_method", "k_ring")),
            return_format="dict",
        )

        # Keep the deformed mesh in its original mesh-local coordinate system.
        # write_augment_obj also writes <sample>_coacd.obj when write_coacd=True.
        augmentor.write_augment_obj(
            output_path=str(paths["final_obj"]),
            write_coacd=write_coacd,
        )

        grasp_yaml_path = write_grasp_yaml(
            paths["grasp"],
            grasp_result=grasp_result,
            mesh_path=paths["final_obj"],
            source_grasp_path=initial_grasp_path,
            top_labels=top_labels,
            sampled_stages=stage_records,
        )

        sample_record = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "seed": int(job["seed"]),
            "top_labels": list(top_labels),
            "expanded_stage_labels": [stage["label"] for stage in sampled_stages],
            "sampled_stages": stage_records,
            "sampling_strategy": job.get("sampling_strategy", "paired"),
            "grid_indices": job.get("grid_indices"),
            "grid_counts": job.get("grid_counts"),
            "combinations_per_chain": job.get("combinations_per_chain"),
            "source_obj_path": obj_path,
            "final_obj_path": str(paths["final_obj"]),
            "raw_obj_path": str(paths["raw_obj"]) if save_raw else None,
            "grasp_path": str(grasp_yaml_path),
            "sample_meta_path": str(paths["sample_meta"]),
            "debug_path": str(paths["debug"]),
            "transferred_grasp": grasp_result,
        }
        sample_meta_path = dump_yaml_or_json(paths["sample_meta"], sample_record)

        debug_path = ""
        if bool(output_cfg.get("save_debug", True)):
            debug_record = {
                "sample": sample_record,
                "mesh_status": augmentor.mesh_status() if hasattr(augmentor, "mesh_status") else None,
                "grasp_anchor": anchor,
                "grasp_transfer_debug": transfer_debug,
                "transferred_grasp": grasp_result,
            }
            debug_path = str(dump_yaml_or_json(paths["debug"], debug_record))

        return {
            "ok": True,
            "sample_id": sample_id,
            "sample_name": sample_name,
            "labels": "+".join(top_labels),
            "stages": " -> ".join(stage["label"] for stage in sampled_stages),
            "obj_path": str(paths["final_obj"]),
            "grasp_path": str(grasp_yaml_path),
            "meta_path": str(sample_meta_path),
            "debug_path": debug_path,
            "error": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "sample_id": sample_id,
            "sample_name": "",
            "labels": "+".join(job.get("top_labels", [])),
            "stages": " -> ".join(
                stage.get("label", "") for stage in job.get("sampled_stages", [])
            ),
            "obj_path": "",
            "grasp_path": "",
            "meta_path": "",
            "debug_path": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# -----------------------------------------------------------------------------
# Batch runner and CLI
# -----------------------------------------------------------------------------


def write_manifest(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "manifest.csv"
    jsonl_path = output_root / "manifest.jsonl"
    fields = [
        "ok", "sample_id", "sample_name", "labels", "stages", "obj_path",
        "grasp_path", "meta_path", "debug_path", "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def apply_cli_overrides(
    meta: Dict[str, Any],
    *,
    obj_path: Optional[str],
    output_root: Optional[str],
) -> Dict[str, Any]:
    result = copy.deepcopy(meta)
    if obj_path is not None:
        result.setdefault("object", {})["obj_path"] = obj_path
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
    labels_per_shape: Optional[int] = None,
    obj_path: Optional[str] = None,
    output_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    meta = apply_cli_overrides(
        load_meta(meta_path),
        obj_path=obj_path,
        output_root=output_root,
    )
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

    output_root_path = Path(meta.get("output", {}).get("root", "fpsa_chained_outputs"))
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
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            futures = [executor.submit(_worker_generate_shape, job) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: int(row.get("sample_id", 10**12)))
    write_manifest(output_root_path, rows)

    succeeded = sum(bool(row.get("ok")) for row in rows)
    print(
        f"[FPSA chained batch] done: {succeeded}/{len(rows)} succeeded; "
        f"manifest: {output_root_path / 'manifest.csv'}"
    )
    if succeeded != len(rows):
        print(
            f"[FPSA chained batch] failed: {len(rows) - succeeded}; "
            f"see {output_root_path / 'manifest.jsonl'}"
        )
    return rows


def split_labels(value: Optional[str]) -> Optional[List[str]]:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def dry_run_summary(jobs: Sequence[Dict[str, Any]], limit: int = 8) -> Dict[str, Any]:
    preview = []
    jobs_per_label = Counter(
        label
        for job in jobs
        for label in job.get("top_labels", [])
    )
    for job in jobs[:limit]:
        preview.append(
            {
                "sample_id": job["sample_id"],
                "seed": job["seed"],
                "top_labels": job["top_labels"],
                "sampling_strategy": job.get("sampling_strategy", "paired"),
                "grid_indices": job.get("grid_indices"),
                "grid_counts": job.get("grid_counts"),
                "stages": [
                    {
                        "label": stage["label"],
                        "method": stage["method"],
                        "linspace_index": stage.get("linspace_index"),
                        "linspace_count": stage.get("linspace_count"),
                        "first_displacement": stage["displacements"][0],
                        "first_magnitude": stage["magnitudes"][0],
                    }
                    for stage in job["sampled_stages"]
                ],
            }
        )
    return {
        "num_jobs": len(jobs),
        "jobs_per_label": dict(jobs_per_label),
        "preview": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiprocessing chained FPSA grasp randomizer for assembly tools"
    )
    parser.add_argument("--meta", required=True, help="Path to chained YAML/JSON meta file")
    parser.add_argument("--labels", default=None, help="Comma-separated top-level chain labels")
    parser.add_argument("--num-shapes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--label-mode",
        choices=["balanced", "random_one", "random_k", "all"],
        default=None,
    )
    parser.add_argument("--labels-per-shape", type=int, default=None)
    parser.add_argument("--obj-path", default=None, help="Override object.obj_path")
    parser.add_argument("--output-root", default=None, help="Override output.root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/expand chains and sampling without importing FPSA",
    )
    args = parser.parse_args()

    labels = split_labels(args.labels)
    if args.dry_run:
        meta = apply_cli_overrides(
            load_meta(args.meta),
            obj_path=args.obj_path,
            output_root=args.output_root,
        )
        jobs = make_jobs(
            meta,
            labels=labels,
            num_shapes=args.num_shapes,
            seed=args.seed,
            label_mode=args.label_mode,
            labels_per_shape=args.labels_per_shape,
        )
        print(json.dumps(to_builtin(dry_run_summary(jobs)), indent=2, ensure_ascii=False))
        return

    run_batch(
        meta_path=args.meta,
        labels=labels,
        num_shapes=args.num_shapes,
        workers=args.workers,
        seed=args.seed,
        label_mode=args.label_mode,
        labels_per_shape=args.labels_per_shape,
        obj_path=args.obj_path,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()