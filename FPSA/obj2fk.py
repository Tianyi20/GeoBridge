import sys
import numpy as np
import igl
from pathlib import Path

def obj_to_fk(obj_path, fk_path=None, radius=5, use_k_ring=True):
    obj_path = Path(obj_path)
    fk_path = Path(fk_path) if fk_path else obj_path.with_suffix(".fk")

    V, F = igl.read_triangle_mesh(str(obj_path))

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"Input mesh must be triangulated, got F.shape={F.shape}")

    result = igl.principal_curvature(V, F, radius, use_k_ring)
    PD1, PD2, PV1, PV2 = result[:4]

    PV1 = np.asarray(PV1, dtype=np.float64).reshape(-1)
    PV2 = np.asarray(PV2, dtype=np.float64).reshape(-1)

    if len(result) >= 5:
        bad_vertices = np.asarray(result[4], dtype=np.int64)
        if bad_vertices.size > 0:
            print(f"Warning: principal_curvature reported {bad_vertices.size} bad vertices")
            # 防止 bad vertices 产生 nan/inf 污染输出
            PV1[bad_vertices] = 0.0
            PV2[bad_vertices] = 0.0

    PV1 = np.nan_to_num(PV1, nan=0.0, posinf=0.0, neginf=0.0)
    PV2 = np.nan_to_num(PV2, nan=0.0, posinf=0.0, neginf=0.0)

    # average_onto_faces 只能处理一维标量，所以 PV1 / PV2 分开做
    face_pv1 = igl.average_onto_faces(F, PV1)
    face_pv2 = igl.average_onto_faces(F, PV2)

    face_curv = np.column_stack([face_pv1, face_pv2])

    if face_curv.shape[0] != F.shape[0]:
        raise RuntimeError(
            f"Row count mismatch: face_curv={face_curv.shape[0]}, faces={F.shape[0]}"
        )

    np.savetxt(fk_path, face_curv, fmt="%.15f %.15f")
    print(f"wrote {fk_path} with {face_curv.shape[0]} rows")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python obj2fk.py input.obj [output.fk]")
        sys.exit(1)

    obj_to_fk(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)