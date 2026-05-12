import numpy as np


class ObjPoseDR(object):
    """Object pose domain randomization."""

    def __init__(self, bullet_client, seed=42):
        self.bullet_client = bullet_client
        self.objpose_rng = np.random.default_rng(seed)
        self.obj_xy_jitter = None
        self.obj_z_axis_rotation_jitter = None

    def sample_obj_pose_randomization(self,
                                      xy_jitter_range=0.2,
                                      z_axis_rotation_range=np.pi):
        rng = self.objpose_rng
        self.obj_xy_jitter = rng.uniform(
            low=[-xy_jitter_range, -xy_jitter_range],
            high=[xy_jitter_range, xy_jitter_range],
        )
        self.obj_z_axis_rotation_jitter = rng.uniform(
            -z_axis_rotation_range,
            z_axis_rotation_range,
        )
        return self.obj_xy_jitter, self.obj_z_axis_rotation_jitter
