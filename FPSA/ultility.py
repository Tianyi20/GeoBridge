import json
import yaml
import os
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np

def load_initial_grasp_pose(path: str):
    """
    读取 grasp_pose.yaml / grasp_pose.json

    返回 dict，包含：
        mesh_path: str
        reference_frame: str
        hand_frame: str
        t: np.ndarray (3,)
        R: np.ndarray (3,3)
        opening: float
        pregrasp_opening: float
        finger_joint: float
        T_mesh_hand: np.ndarray (4,4)
        T_mesh_hand_tcp: np.ndarray (4,4)

    优先级：
      1. 若文件中有 T_mesh_hand，则直接使用它恢复 t 和 R
      2. 否则使用 position_m + quaternion_xyzw
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"抓取文件不存在: {path}")

    suffix = Path(path).suffix.lower()
    if suffix in [".yaml", ".yml"]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {Path(path).suffix}，只支持 .yaml/.yml/.json")

    mesh_path = str(data.get("mesh_path", ""))
    reference_frame = str(data.get("reference_frame", "mesh_local_frame"))
    hand_frame = str(data.get("hand_frame", "panda_hand"))
    opening = float(data.get("opening_width_m", 0.06))
    pregrasp_opening = float(data.get("pregrasp_opening_width_m", opening))
    finger_joint = float(data.get("finger_joint_m", opening / 2.0))

    if "T_mesh_hand" in data and data["T_mesh_hand"] is not None:
        T_mesh_hand = np.asarray(data["T_mesh_hand"], dtype=float)
        if T_mesh_hand.shape != (4, 4):
            raise ValueError("T_mesh_hand 维度错误，应为 4x4")

    if "T_mesh_hand_tcp" in data and data["T_mesh_hand_tcp"] is not None:
        T_mesh_hand_tcp = np.asarray(data["T_mesh_hand_tcp"], dtype=float)
        if T_mesh_hand_tcp.shape != (4, 4):
            raise ValueError("T_mesh_hand_tcp 维度错误，应为 4x4")
        R = T_mesh_hand_tcp[:3, :3]
        t = T_mesh_hand_tcp[:3, 3]
        quat = Rotation.from_matrix(R).as_quat()

    return {
        "mesh_path": mesh_path,
        "reference_frame": reference_frame,
        "hand_frame": hand_frame,
        "t": t,
        "R": R,
        "quat": quat,
        "opening": opening,
        "pregrasp_opening": pregrasp_opening,
        "finger_joint": finger_joint,
        "T_mesh_hand": T_mesh_hand,
        "T_mesh_hand_tcp": T_mesh_hand_tcp,
    }
