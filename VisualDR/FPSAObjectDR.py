from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class FPSAObjectDR:
    """Sample a mesh and its corresponding grasp pose from FPSA outputs.

    The sample pool contains:
      1. the base mesh + base grasp pose, when provided;
      2. every FPSA augmented .obj that has a matching *_grasp.yaml / *_grasp.yml.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(int(seed))
        self.sample_pool: List[Dict[str, str]] = []
        self.last_sample: Optional[Dict[str, str]] = None

    @staticmethod
    def _as_path(path: str) -> Path:
        return Path(path).expanduser()

    @staticmethod
    def _matching_grasp_path(obj_path: Path) -> Optional[Path]:
        for suffix in (".yaml", ".yml"):
            grasp_path = obj_path.with_name(f"{obj_path.stem}_grasp{suffix}")
            if grasp_path.exists():
                return grasp_path
        return None

    def build_sample_pool(
        self,
        base_mesh_path: Optional[str] = None,
        base_grasp_path: Optional[str] = None,
        fpsa_aug_root: Optional[str] = None,
        include_base: bool = True,
    ) -> List[Dict[str, str]]:
        pool: List[Dict[str, str]] = []

        if include_base and (base_mesh_path is not None or base_grasp_path is not None):
            if base_mesh_path is None or base_grasp_path is None:
                raise ValueError("base_mesh_path and base_grasp_path must be provided together")

            base_mesh = self._as_path(base_mesh_path)
            base_grasp = self._as_path(base_grasp_path)
            if not base_mesh.exists():
                raise FileNotFoundError(f"Base mesh not found: {base_mesh}")
            if not base_grasp.exists():
                raise FileNotFoundError(f"Base grasp pose not found: {base_grasp}")

            pool.append({
                "obj_path": str(base_mesh),
                "grasp_path": str(base_grasp),
                "source": "base",
            })

        if fpsa_aug_root is not None:
            root = self._as_path(fpsa_aug_root)
            if not root.exists():
                raise FileNotFoundError(f"FPSA augmented object folder not found: {root}")

            for obj_path in sorted(root.glob("*.obj")):
                grasp_path = self._matching_grasp_path(obj_path)
                if grasp_path is None:
                    continue
                pool.append({
                    "obj_path": str(obj_path),
                    "grasp_path": str(grasp_path),
                    "source": "fpsa_aug",
                })

        if len(pool) == 0:
            raise FileNotFoundError("No valid object/grasp pair found for FPSA object randomization")

        self.sample_pool = pool
        return pool

    def sample(
        self,
        base_mesh_path: Optional[str] = None,
        base_grasp_path: Optional[str] = None,
        fpsa_aug_root: Optional[str] = None,
        include_base: bool = True,
    ) -> Tuple[str, str]:
        pool = self.build_sample_pool(
            base_mesh_path=base_mesh_path,
            base_grasp_path=base_grasp_path,
            fpsa_aug_root=fpsa_aug_root,
            include_base=include_base,
        )
        sample = pool[int(self.rng.integers(len(pool)))]
        self.last_sample = dict(sample)
        return sample["obj_path"], sample["grasp_path"]

    def sample_FPSA_augmented_object(
        self,
        base_mesh_path: Optional[str] = None,
        base_grasp_path: Optional[str] = None,
        fpsa_aug_root: Optional[str] = None,
        include_base: bool = True,
    ) -> Tuple[str, str]:
        return self.sample(
            base_mesh_path=base_mesh_path,
            base_grasp_path=base_grasp_path,
            fpsa_aug_root=fpsa_aug_root,
            include_base=include_base,
        )
