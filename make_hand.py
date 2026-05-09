import copy
import numpy as np
import open3d as o3d


def make_transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    x, y, z = xyz
    rr, rp, ry = rpy

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rr), -np.sin(rr)],
        [0, np.sin(rr),  np.cos(rr)],
    ])
    Ry = np.array([
        [ np.cos(rp), 0, np.sin(rp)],
        [0,           1, 0],
        [-np.sin(rp), 0, np.cos(rp)],
    ])
    Rz = np.array([
        [np.cos(ry), -np.sin(ry), 0],
        [np.sin(ry),  np.cos(ry), 0],
        [0,           0,          1],
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z])
    return T


def load_franka_hand_opened(
    hand_mesh_path,
    finger_mesh_path,
    open_width=0.04,
):
    # 读 mesh
    hand = o3d.io.read_triangle_mesh(hand_mesh_path)
    left_finger = o3d.io.read_triangle_mesh(finger_mesh_path)
    right_finger = o3d.io.read_triangle_mesh(finger_mesh_path)

    # 法向量，便于渲染
    for m in (hand, left_finger, right_finger):
        m.compute_vertex_normals()

    # ===== panda_hand =====
    # hand 自己不动，作为基座

    # ===== panda_leftfinger =====
    # joint origin: xyz = (0, 0, 0.0584)
    # axis = (0, 1, 0)
    # q = open_width
    T_left = make_transform(xyz=(0, open_width, 0.0584), rpy=(0, 0, 0))
    left_finger.transform(T_left)

    # ===== panda_rightfinger =====
    # visual origin 还有一个 rpy=(0, 0, pi)
    # joint axis = (0, -1, 0)
    T_right = make_transform(xyz=(0, -open_width, 0.0584), rpy=(0, 0, 0))
    T_right_visual = make_transform(xyz=(0, 0, 0), rpy=(0, 0, np.pi))
    right_finger.transform(T_right @ T_right_visual)

    return hand, left_finger, right_finger


if __name__ == "__main__":
    hand, left_finger, right_finger = load_franka_hand_opened(
        hand_mesh_path="./franka_panda/meshes/visual/hand.obj",
        finger_mesh_path="./franka_panda/meshes/visual/finger.obj",
        open_width=0.04,   # 最大张开
    )

    o3d.visualization.draw_geometries(
        [hand, left_finger, right_finger],
        window_name="Franka Panda Hand (Opened)",
        mesh_show_back_face=True,
    )
    combined = hand + left_finger + right_finger
    o3d.io.write_triangle_mesh("franka_hand_opened.obj", combined)
