import numpy as np
import pybullet_data
from pybullet_utils import bullet_client as bc
from PickUpEnv import PickUpEnv
import pybullet as p


env = PickUpEnv(
    sim_steps_per_action=10,
    connection_mode= p.GUI,
    seed=42,
)

obs = env.reset()

print("reset ok")
print("obs keys:", obs.keys())
print("image:", obs["agentview_image"].shape, obs["agentview_image"].dtype)
print("eef pos:", obs["robot0_eef_pos"])
print("eef quat:", obs["robot0_eef_quat"])
print("gripper:", obs["robot0_gripper_qpos"])
print("obs in space:", env.observation_space.contains(obs))

for i in range(1000):
    # 用当前 eef pose 做一个安全 action，不乱采样
    action = np.concatenate([
        obs["robot0_eef_pos"] + np.array([0.0, 0.0, 0.02], dtype=np.float32),   # 上升 2cm
        obs["robot0_eef_quat"],
        np.array([0.04], dtype=np.float32),
    ]).astype(np.float32)

    # print(action)
    obs, reward, done, info = env.step(action)
    print(f"step {i}: reward={reward}, done={done}")

img = env.render()
print("render:", img.shape, img.dtype)

env.close()
print("close ok")