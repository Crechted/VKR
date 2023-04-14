import gym
from HER.rl_modules.ddpg_agent import ddpg_agent
from mpi4py import MPI
import random
import torch
import numpy as np
import os
from configs.env_info import get_HER_cfgs
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def launch(args, env):
    # set random seeds for reproduce
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    seed = args.seed + MPI.COMM_WORLD.Get_rank()
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    # get the environment parameters
    env_params = get_HER_cfgs(env)
    print('\nENV PARAMS: ', env_params)
    print('SEED: ', seed)
    print('SAMPLE ENV: ', env.reset())
    print(f'Epochs: {args.n_epochs}, episodes: {args.n_epochs*args.n_cycles*args.num_rollouts_per_mpi}, steps: {args.n_epochs*args.n_cycles*args.num_rollouts_per_mpi*env._max_episode_steps}')

    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()
