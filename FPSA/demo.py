import numpy as np
import slippage_reshaping as sr
from pathlib import Path
import igl
import trimesh
from icecream import ic

def get_fk(obj_path, radius=5, use_k_ring=True):
    obj_path = Path(obj_path)

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
            PV1[bad_vertices] = 0.0
            PV2[bad_vertices] = 0.0

    PV1 = np.nan_to_num(PV1, nan=0.0, posinf=0.0, neginf=0.0)
    PV2 = np.nan_to_num(PV2, nan=0.0, posinf=0.0, neginf=0.0)

    face_pv1 = igl.average_onto_faces(F, PV1)
    face_pv2 = igl.average_onto_faces(F, PV2)

    face_curv = np.column_stack([face_pv1, face_pv2])

    if face_curv.shape[0] != F.shape[0]:
        raise RuntimeError(
            f"Row count mismatch: face_curv={face_curv.shape[0]}, faces={F.shape[0]}"
        )

    return face_curv


def load_mesh_and_face_curvature(obj_path):
    V, F = igl.read_triangle_mesh(str(obj_path))

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    face_curv = get_fk(obj_path)
    face_k1 = np.asarray(face_curv[:, 0], dtype=np.float64).reshape(-1)
    face_k2 = np.asarray(face_curv[:, 1], dtype=np.float64).reshape(-1)

    if face_k1.shape[0] != F.shape[0] or face_k2.shape[0] != F.shape[0]:
        raise RuntimeError(
            f"Curvature size mismatch: face_k1={face_k1.shape}, "
            f"face_k2={face_k2.shape}, F={F.shape}"
        )

    return V, F, face_k1, face_k2


def write_obj(path, V, F):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # OBJ indices are 1-based
        for tri in F:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


if __name__ == "__main__":
    obj_path = "/home/iadc/GeoBridge/data/objects/mug/original_mug/model.obj"

    trimesh_obj = trimesh.load(obj_path, process=False)
    print("trimesh loaded:", obj_path)
    print("trimesh vertices:", trimesh_obj.vertices.shape)
    print("trimesh faces:", trimesh_obj.faces.shape)

    V, F, face_k1, face_k2 = load_mesh_and_face_curvature(obj_path)

    print("igl V:", V.shape)
    print("igl F:", F.shape)
    print("face_k1:", face_k1.shape)
    print("face_k2:", face_k2.shape)

    # Constraint vertices.
    # 其他 constraint ids 保持原位，相当于 fixed constraints。
    constraint_ids = [1087, 1135, 1260, 1276, 3551]

    # sr.optimize_mesh 需要 absolute target positions，不是 displacement。

    bbox_diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    ic(bbox_diag)
    target_positions = V[constraint_ids].copy()

    displacement_3551 = np.array([
        0.0067756052628117945,
        -1.566278750761965e-08,
        -0.1021735721360982,
    ])

    idx = constraint_ids.index(3551)
    target_positions[idx] = V[3551] + displacement_3551 * bbox_diag

    print("\nConstraints:")
    for vid, target in zip(constraint_ids, target_positions):
        print(f"  vid={vid}, original={V[vid]}, target={target}, disp={target - V[vid]}")

    V_opt = sr.optimize_mesh(
        V,
        F,
        face_k1,
        face_k2,
        constraint_ids,
        target_positions,
        max_iters=20,
        handle_error_distrib_enabled=False,
        input_name="cutlery_array_test",
    )

    print("\nOptimization done")
    print("V_opt:", V_opt.shape)

    input_obj = f"./input.obj"
    output_obj = f"./reshaped.obj"

    write_obj(input_obj, V, F)
    write_obj(output_obj, V_opt, F)

    print("\nWrote:")
    print(" ", input_obj)
    print(" ", output_obj)