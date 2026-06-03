from matplotlib.pylab import sample
import wandb
import numpy as np
import torch
import collections
import math
import pybullet as p

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply
from PickUpEnv import PickUpEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from icecream import ic
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

class PickUpEnvDebugRunner(BaseImageRunner):
    def __init__(self, output_dir,            
                 max_steps=2000,
                 _seed = 43,
                 randomize_image_noise = True,
                 randomize_lighting = True,
                 randomize_objpose = True,
                 randomize_distractors = True,
                 randomize_outlscene = True,
                 randomize_plane_height = True,
                 randomize_campose = True,
                 n_obs_steps=2,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 video_file_path = None, # None to disable video recording
                 ):
        super().__init__(output_dir)

        self.n_action_steps = n_action_steps
        ic(n_obs_steps, n_action_steps)
        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PickUpEnv(
                        sim_steps_per_action=12,
                        connection_mode=p.GUI,
                        seed = _seed,
                        randomize_image_noise=randomize_image_noise,
                        randomize_lighting=randomize_lighting,
                        randomize_objpose=randomize_objpose,
                        randomize_distractors=randomize_distractors,
                        randomize_outlscene = randomize_outlscene,
                        randomize_plane_height = randomize_plane_height,
                        randomize_campose = randomize_campose,
                        ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path= video_file_path,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        self.env = env_fn()

    def _prepare_obs(self, obs):
        np_obs_dict = dict(obs)

        # image: uint8 NHWC -> float32 NCHW, range [0, 1]
        for key in ['agentview_image']:
            if key in np_obs_dict:
                np_obs_dict[key] = np.moveaxis(
                    np_obs_dict[key], -1, -3
                ).astype(np.float32) / 255.0

        for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
            if key in np_obs_dict:
                np_obs_dict[key] = np_obs_dict[key].astype(np.float32)

        return np_obs_dict


    def run(self, policy: BaseImagePolicy):
        device = policy.device

        obs = self.env.reset()
        policy.reset()

        done = False
        while not done:
            np_obs_dict = self._prepare_obs(obs)

            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device)
            )
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            np_action_dict = dict_apply(
                action_dict,
                lambda x: x.detach().cpu().numpy()
            )

            action = np_action_dict['action']   # (1, 8, 7)
            action = action[:, :self.n_action_steps, :].squeeze(0)
            ic(action.shape)
            # 期望形状: (n_envs, n_action_steps, action_dim)
            obs, reward, done, info = self.env.step(action)
            done = np.all(done)

        return