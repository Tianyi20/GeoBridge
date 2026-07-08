import collections
import math
import pathlib

import dill
import numpy as np
import pybullet as p
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from WrenchFisheyeEnv import WrenchEnv
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from icecream import ic

class WrenchEngagementEnvRunner(BaseImageRunner):
    def __init__(self, output_dir,            
                 max_steps=2000,
                 n_obs_steps=2,
                 n_action_steps=8,
                 use_agent_cam = True,
                 use_fisheye_cam = True,
                 # How many envs as env runners
                 train_start_seed=43,
                 test_start_seed=20043,
                 n_train=10,
                 n_train_vis=3,
                 n_test=12,
                 n_test_vis=6,
                 fps=10,
                 crf=22,
                 video_file_path = None, # None to disable video recording
                 n_envs=None,
                 tqdm_interval_sec=5.0,
                 ):
        super().__init__(output_dir)

        ic(n_obs_steps, n_action_steps)
        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    WrenchEnv(
                        sim_steps_per_action=12,
                        connection_mode=p.DIRECT,
                        seed = 46,
                        use_agent_cam= use_agent_cam,
                        use_fisheye_cam= use_fisheye_cam,
                        if_FPSA = False,
                        randomize_objcolor = True,
                        randomize_image_noise=True,
                        randomize_lighting=True,
                        randomize_objpose=True,
                        randomize_distractors=True,
                        randomize_outlscene = True,
                        randomize_plane_height = True,
                        randomize_campose = True
                        ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path= None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        if n_envs is None:
            n_envs = n_train + n_test
        
        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train env init
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # env: MultiStepWrapper
                # env.env: VideoRecordingWrapper
                assert isinstance(env, MultiStepWrapper)
                assert isinstance(env.env, VideoRecordingWrapper)

                # 先停掉上一次录像，避免 buffer / 文件句柄残留
                env.env.video_recoder.stop()
                env.env.file_path = None

                # 只有 enable_render=True 的 episode 才真正写 mp4
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + '.mp4'
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)

                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test env init
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env, MultiStepWrapper)
                assert isinstance(env.env, VideoRecordingWrapper)

                env.env.video_recoder.stop()
                env.env.file_path = None

                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + '.mp4'
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)

                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, shared_memory=False)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def _prepare_obs(self, obs):
        np_obs_dict = dict(obs)

        # image: uint8 NHWC -> float32 NCHW, range [0, 1]
        for key in ['agentview_image', "robot0_eye_in_hand_image"]:
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
        env = self.env

        # rollout plan
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # buffers
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs: set seed + decide whether to render to file
            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=(
                    f'Eval PickUpEnv chunk: {chunk_idx + 1}/{n_chunks}\n'
                    f'| Current chunk contains: ({this_n_active_envs} envs)'
                    f"| Total env eval={n_inits}"
                ),
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done:
                np_obs_dict = self._prepare_obs(obs)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to('cpu').numpy(),
                )
                action = np_action_dict['action']

                # 期望形状: (n_envs, n_action_steps, action_dim)
                obs, reward, done, info = env.step(action)
                done = np.all(done)

                # reward 一般是每个 env 的时间序列累计在 wrapper 内部，
                # 这里进度按 rollout 执行的 action horizon 前进
                if action.ndim >= 2:
                    pbar.update(action.shape[1])
                else:
                    pbar.update(1)
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # clear internal video buffers
        # _ = env.reset()
        # logging
        max_rewards = collections.defaultdict(list)
        log_data = {}

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = float(np.max(all_rewards[i]))
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + f'sim_video_{seed}'] = wandb.Video(video_path)

        for prefix, value in max_rewards.items():
            mean_score = float(np.mean(value))
            log_data[prefix + 'mean_score'] = mean_score

            # 给 checkpoint monitor 用，避免 key 里有 /
            clean_prefix = prefix.rstrip('/').replace('/', '_')
            log_data[clean_prefix + '_mean_score'] = mean_score
        ic(log_data)
        return log_data