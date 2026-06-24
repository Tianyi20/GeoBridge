from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import numpy as np


tf = RotationTransformer(
from_rep='quaternion',
to_rep='matrix'
)

# Case 1: wxyz identity quaternion
q_wxyz_identity = np.array([[1., 0., 0., 0.]], dtype=np.float32)
mat_from_wxyz = tf.forward(q_wxyz_identity)

# Case 2: xyzw identity quaternion
q_xyzw_identity = np.array([[0., 0., 0., 1.]], dtype=np.float32)
mat_from_xyzw = tf.forward(q_xyzw_identity)

I = np.eye(3, dtype=np.float32)[None]

print("mat_from_wxyz:")
print(mat_from_wxyz)

print("mat_from_xyzw:")
print(mat_from_xyzw)

print("wxyz identity error:", np.abs(mat_from_wxyz - I).max())
print("xyzw identity error:", np.abs(mat_from_xyzw - I).max())

assert np.allclose(mat_from_wxyz, I, atol=1e-6)
assert not np.allclose(mat_from_xyzw, I, atol=1e-6)

print("Conclusion: PyTorch3D / RotationTransformer expects quaternion in wxyz order.")