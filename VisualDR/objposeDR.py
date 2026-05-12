import numpy as np


class ObjPoseDR(object):
    """Object pose domain randomization."""

    def __init__(self, bullet_client, seed=42):
        self.bullet_client = bullet_client
        self.objpose_rng = np.random.default_rng(seed)
        self.obj_xy_jitter = None
        self.obj_z_axis_rotation_jitter = None

    def sample_obj_pose_randomization(
        self,
        x_jitter_range=0.2,
        y_jitter_range=0.2,
        z_axis_rotation_range=np.pi,
    ):
        rng = self.objpose_rng

        obj_x_jitter = rng.uniform(-x_jitter_range, x_jitter_range)
        obj_y_jitter = rng.uniform(-y_jitter_range, y_jitter_range)

        self.obj_xy_jitter = np.array(
            [obj_x_jitter, obj_y_jitter],
            dtype=np.float32,
        )

        self.obj_z_axis_rotation_jitter = rng.uniform(
            -z_axis_rotation_range,
            z_axis_rotation_range,
        )

        return self.obj_xy_jitter, self.obj_z_axis_rotation_jitter