import os
import json
import coacd
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float32)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float32)

def quat_to_rotvec(q):
    q = np.asarray(q, dtype=np.float32)
    q = q / np.linalg.norm(q)
    xyz = q[:3]
    w = np.clip(q[3], -1.0, 1.0)

    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis = xyz / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    return (axis * angle).astype(np.float32)

def quat_diff_rotvec(q_from, q_to):
    # q_delta = q_to * conj(q_from)
    q_delta = quat_multiply(
        np.asarray(q_to, dtype=np.float32),
        quat_conjugate(np.asarray(q_from, dtype=np.float32))
    )
    # shortest path
    if q_delta[3] < 0:
        q_delta = -q_delta
    return quat_to_rotvec(q_delta)

def quat_slerp(q0, q1, t):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # 解决双覆盖，走最短路径
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)

    return q0 * np.cos(theta) + q2 * np.sin(theta)


def load_models(bullet_client,
                visual_mesh_file, 
                vhacd_mesh_file, 
                desired_mass = 1.0,
                position = [0.0,0.0,0.0],
                baseOrientation = [0,0,0,1],
                center_of_mass = [0.0,0.0,0.0],
                lateral_friction = 0.6,
                spinning_friction = 0.003,
                visual_only=False,
                ):
    """
    Load a body into the simulation, given the visual mesh file and the vhacd mesh file.
    Position should be zero if the mesh's origin is at world origin.
    Center of mass very important, should be calculated before loading.
    Returns the body id of the loaded body.
    """
    visual_shape_id = bullet_client.createVisualShape(shapeType=bullet_client.GEOM_MESH,
                                          fileName=visual_mesh_file,
                                          meshScale=[1, 1, 1])
    # Optional collision shape
    if visual_only:
        collision_shape_id = -1
    else:
        assert vhacd_mesh_file is not None, \
            "vhacd_mesh_file is required unless visual_only=True"
        collision_shape_id = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=vhacd_mesh_file,
            meshScale=[1, 1, 1],
        )
    
    body_id = bullet_client.createMultiBody(baseMass= desired_mass,  # 你可以根据需要指定质量
                                baseCollisionShapeIndex=collision_shape_id,
                                baseVisualShapeIndex=visual_shape_id,
                                basePosition=position,
                                # 这里的 baseInertialFramePosition 是 center of mass 相对于 baseposition. very importatnt property. 的pose。-0.01743, -0.08127,  0.02427
                                baseInertialFramePosition=center_of_mass, 
                                #baseInertialFrameOrientation=[0, 0, 0, 1],
                                baseOrientation=baseOrientation)
    if not visual_only:
        bullet_client.changeDynamics(body_id, -1, lateralFriction = lateral_friction, 
                            spinningFriction = spinning_friction,
                            restitution = 0.5,
                            rollingFriction = 0.005)

    return body_id

def coacd_convex_decomposition(obj_filename,threshold = 0.03, preprocess_resolution = 100):
    """
    COACD convex decomposition
    """
    output_file = os.path.splitext(obj_filename)[0] + "_coacd.obj"

    if os.path.exists(output_file):
        print(f"[COACD]: Convex decomposition mesh {output_file} exists")
        return output_file
    
    mesh = trimesh.load(obj_filename, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    #result = coacd.run_coacd(mesh, threshold= 0.02, mcts_max_depth= 5, mcts_nodes= 30, preprocess_mode= "False") # a list of convex hulls.
    result = coacd.run_coacd(mesh, threshold = threshold, preprocess_resolution =preprocess_resolution) # a list of convex hulls.

    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    scene.export(output_file)

    return output_file

def get_com(mesh_file):
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Input file '{mesh_file}' does not exist.")

    return trimesh.load(mesh_file).bounding_box.centroid

def get_true_PositionAndOrientation(bullet_client, body_id):

    dyn = bullet_client.getDynamicsInfo(body_id, -1)
    local_inertial_pos = dyn[3]      # baseInertialFramePosition
    local_inertial_orn = dyn[4]      # baseInertialFrameOrientation

    # 2) 刚体（返回的是惯性/COM坐标系）在世界下的位姿
    com_world_pos, com_world_orn = bullet_client.getBasePositionAndOrientation(body_id)

    # 3) 反求“mesh/基坐标系”的世界位姿：  T^W_base = T^W_com * (T^base_inertial)^(-1)
    inv_pos, inv_orn = bullet_client.invertTransform(local_inertial_pos, local_inertial_orn)
    mesh_world_pos, mesh_world_orn = bullet_client.multiplyTransforms(com_world_pos, com_world_orn,
                                                        inv_pos, inv_orn)
    
    return mesh_world_pos, mesh_world_orn

def get_COM_PositionAndOrientation(bullet_client, body_id):

    com_world_pos, com_world_orn = bullet_client.getBasePositionAndOrientation(body_id)
    return com_world_pos, com_world_orn

    
def cvK2BulletP(K, w, h, near, far):
    """
    cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
    and ROS to the projection matrix used in openGL and Pybullet.

    :param K:  OpenCV 3x3 camera intrinsic matrix
    :param w:  Image width
    :param h:  Image height
    :near:     The nearest objects to be included in the render
    :far:      The furthest objects to be included in the render
    :return:   4x4 projection matrix as used in openGL and pybullet
    """ 
    f_x = K[0,0]
    f_y = K[1,1]
    c_x = K[0,2]
    c_y = K[1,2]
    A = (near + far)/(near - far)
    B = 2 * near * far / (near - far)

    projection_matrix = [
                        [2/w * f_x,  0,          (w - 2*c_x)/w,  0],
                        [0,          2/h * f_y,  (2*c_y - h)/h,  0],
                        [0,          0,          A,              B],
                        [0,          0,          -1,             0]]
    #The transpose is needed for respecting the array structure of the OpenGL
    return np.array(projection_matrix).T.reshape(16).tolist()


def cvPose2BulletView(T):
    """
    cvPose2BulletView gets orientation and position as used 
    in ROS-TF and opencv and coverts it to the view matrix used 
    in openGL and pyBullet.
    
    :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw] 
    :param t: ROS postion expressed as [tx, ty, tz]
    :return:  4x4 view matrix as used in pybullet and openGL
    
    """

    T = T
    # Convert opencv convention to python convention
    # By a 180 degrees rotation along X
    Tc = np.array([[1,   0,    0,  0],
                    [0,  -1,    0,  0],
                    [0,   0,   -1,  0],
                    [0,   0,    0,  1]]).reshape(4,4)
    
    # pybullet pse is the inverse of the pose from the ROS-TF
    T=Tc@np.linalg.inv(T)
    # The transpose is needed for respecting the array structure of the OpenGL
    viewMatrix = T.T.reshape(16)
    return viewMatrix
