import torch
from TD_MPC.src.algorithm.tdmpc import TDMPC

import numpy as np
from configs.env_info import get_TDMPC_cfgs
from TD_MPC.src.algorithm.helper import Episode, ReplayBuffer
from TD_MPC.train import set_seed, get_obs, evaluate
import os
import time


# python main.py --demo --env-name='reach' --render

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


def launch(args, env):
    assert torch.cuda.is_available()

    cwd = os.getcwd()
    s_name = '/model_mu.pt' if args.env_name == 'mujoco' else '/model_py.pt'
    model_path = cwd + '/TD_MPC/saved_models/' + args.task_name + s_name
    # create the environment
    cfg = get_TDMPC_cfgs(args, env)
    set_seed(cfg.seed)
    agent, buffer = TDMPC(cfg), ReplayBuffer(cfg)
    # get the environment params
    agent.load(model_path)
    for ep in range(cfg.demo_length):
        # Collect trajectory
        step = ep*cfg.episode_length
        obs = env.reset()
        episode = Episode(cfg, get_obs(obs))
        while not episode.done:
            if args.env_name == 'mujoco' and args.render:
                env.render()
            action = agent.plan(get_obs(obs), eval_mode=True, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (get_obs(obs), action, reward, done)
            time.sleep(0.02)
        assert len(episode) == cfg.episode_length
        buffer += episode
        print(f'the episode is: {int(step/cfg.episode_length)}, reward: {episode.cumulative_reward}')
