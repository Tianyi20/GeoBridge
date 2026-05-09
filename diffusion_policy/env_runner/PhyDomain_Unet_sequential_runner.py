from matplotlib.pylab import sample
import wandb
import numpy as np
import torch
import collections
import math
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply
from CupOnRackEnv import CupOnRackEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from icecream import ic
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

class PhyDomainImageRunner(BaseImageRunner):
    def __init__(self, output_dir,            
                 max_steps=2000,
                 n_obs_steps=2,
                 n_action_steps=2,
                 fps=10,
                 crf=22,
                 video_file_path = None, # None to disable video recording
                 ):
        super().__init__(output_dir)

        ic(n_obs_steps, n_action_steps)
        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    CupOnRackEnv(sim_steps_per_action=12),
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

    def run(self, policy: BaseImagePolicy):
        device = policy.device

        obs = self.env.reset()
        policy.reset()

        done = False
        while not done:
            # obs(dict of numpy) -> torch
            np_obs_dict = dict(obs)
            np_obs_dict['agentview_image'] = np.moveaxis(
                np_obs_dict['agentview_image'], -1, 1
            ).astype(np.float32) / 255.0
            np_obs_dict['robot0_eye_in_hand_image'] = np.moveaxis(
                np_obs_dict['robot0_eye_in_hand_image'], -1, 1
            ).astype(np.float32) / 255.0
            np_obs_dict['robot0_eef_pos'] = np_obs_dict['robot0_eef_pos'].astype(np.float32)
            np_obs_dict['robot0_eef_quat'] = np_obs_dict['robot0_eef_quat'].astype(np.float32)
            np_obs_dict['robot0_gripper_qpos'] = np_obs_dict['robot0_gripper_qpos'].astype(np.float32)

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

            raw_action = np_action_dict['action']   # (1, 8, 7)
            print("raw_action.shape:", raw_action.shape)
            print("raw_action[0,0]:", raw_action[0, 0])

            # 先只执行前2步
            obs, reward, done, info = self.env.step(raw_action[:, :2, :].squeeze(0))

        return