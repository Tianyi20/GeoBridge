import gym
import numpy as np
import pybullet as p
from gym.envs.registration import register

register(
    id="PickUp-v0",
    entry_point="PickUpEnv:PickUpEnv",
)

env = gym.make(
    "PickUp-v0",
    connection_mode=p.GUI,
    sim_steps_per_action=10,
    seed=42,
    randomize_image_noise=False,
    randomize_lighting=False,
    randomize_objpose=False,
    randomize_distractors=False,
)

obs = env.reset()
print("reset ok")
print("image:", obs["agentview_image"].shape, obs["agentview_image"].dtype)
print("obs in space:", env.observation_space.contains(obs))

for i in range(100):
    action = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        np.array([0.04], dtype=np.float32),
    ]).astype(np.float32)

    obs, reward, done, info = env.step(action)
    print(i, reward, done)

    if done:
        break

env.close()
print("close ok")