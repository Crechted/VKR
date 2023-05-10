import argparse
import os
import pickle
import sys
import time

from LOGO.utils import torch
from LOGO.utils.torch import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LOGO.utils.delay_env import DelayRewardWrapper
from LOGO.models.mlp_policy import Policy
from LOGO.models.mlp_critic import Value
from LOGO.models.mlp_policy_disc import DiscretePolicy
from LOGO.models.mlp_discriminator import Discriminator
from torch import nn
from LOGO.core.trpo import trpo_step
from LOGO.core.common import estimate_advantages
from LOGO.core.agent import Agent
# from torch.utils.tensorboard import SummaryWriter
from collections import deque
import configs.parse_csv as par
from datetime import datetime

args, discrim_net, writer, device = None, None, None, None
value_net, value_net_exp, policy_net, partial_expert_traj = None, None, None, None
optimizer_discrim, discrim_criterion, dtype, render = None, None, None, False



def create_model_path(cfgs):
    if not os.path.exists(cfgs.save_dir):
        os.mkdir(cfgs.save_dir)
    # path to save the model
    model_path = os.path.join(cfgs.save_dir, cfgs.task_name)
    model_path += f"/{cfgs.noise_eps}" \
                  f"_{cfgs.random_eps}" \
                  f"_{cfgs.replay_k}" \
                  f"_{cfgs.clip_obs}" \
                  f"_{cfgs.batch_size}"\
                  f"_{cfgs.gamma}" \
                  f"_{cfgs.action_l2}" \
                  f"_{cfgs.lr_actor}" \
                  f"_{cfgs.lr_critic}" \
                  f"_{cfgs.polyak}" \
                  f"_{cfgs.seed}"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    return model_path


def launch(cfg, env):
    global args, discrim_net, writer, device, value_net, value_net_exp, policy_net, partial_expert_traj, optimizer_discrim, discrim_criterion, dtype, render

    args = cfg
    args.model_path = create_model_path(args)
    eval_env = env
    render = args.render
    env = DelayRewardWrapper(env, args.delay_val, 1000)

    nn_size = tuple(args.nn_param)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    # writer = SummaryWriter(f'LOGO/Results/{args.task_name}/')
    if args.K_delta > -1:
        print('Adaptive Decay')
        print('delta_0:', args.delta_0)
        print('Warmup Iterations:', args.K_delta)
        print('KL geometric decay value:', args.delta)
    else:
        print('Constant Decay')
        print('delta_0:', args.delta_0)

    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
        print('Using cuda device:', device)

    obs_shape = tuple(
        x + y for x in env.observation_space['observation' if args.env_name == 'mujoco' else 'state'].shape for y in env.observation_space['desired_goal'].shape)
    if args.observe == 0:
        # args.observe = env.observation_space.shape[0]
        args.observe = obs_shape[0]

    print('Observing the first ' + str(args.observe) + ' states')
    # state_dim = env.observation_space.shape[0]
    state_dim = obs_shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    action_dim = 1 if is_disc_action else env.action_space.shape[0]

    """seeding"""

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    """define actor and critic"""
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std, hidden_size=nn_size)

    value_net = Value(state_dim, hidden_size=nn_size)
    value_net_exp = Value(state_dim, hidden_size=nn_size)
    discrim_net = Discriminator(args.observe + action_dim)
    discrim_criterion = nn.BCELoss()

    to_device(device, policy_net, value_net, value_net_exp, discrim_net, discrim_criterion)

    optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

    # load trajectory
    # expert_traj = pickle.load(open(args.demo_traj_path, "rb"))
    expert_traj = np.random.rand(args.min_batch_size, state_dim+action_dim)
    action_indices = [i for i in range(state_dim, expert_traj.shape[1])]
    state_indices = [i for i in range(args.observe)]
    state_action_indices = state_indices + action_indices
    partial_expert_traj = expert_traj[:, state_action_indices]
    print('Demo trajectory samples: ', partial_expert_traj.shape[0])

    """create agent"""
    agent = Agent(args, env, policy_net, device, eval_env=eval_env,
                  num_threads=args.num_threads)

    # writer.add_text('Evaluation env name', str(args.task_name))
    # writer.add_text('Demonstration trajectories path', str(args.demo_traj_path))
    # writer.add_text('K_delta', str(args.K_delta))
    # writer.add_text('delta_0', str(args.delta_0))
    # writer.add_text('delta', str(args.delta))
    # writer.add_text('Seed', str(args.seed))
    # writer.add_text('Observable state', str(args.observe))
    # writer.add_text('Expert trajectory samples', str(partial_expert_traj.shape))
    # if args.model_path is not None:
        # writer.add_text('Model Path', str(args.model_path))

    main_loop(agent)
    _, log_eval = agent.collect_samples(2000, ep_len=args.episode_length, eval_flag=True,
                                        mean_action=True, render=render)
    return log_eval['avg_reward']




##########################################################################
def demo_reward(state, action):
    partial_state = state[:, :args.observe]
    partial_state_action = tensor(np.hstack([partial_state, action]), dtype=dtype).to(device)
    with torch.no_grad():
        return -torch.log(discrim_net(partial_state_action)).squeeze()


def update_params(batch, i_iter, kl):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    rewards_exp = demo_reward(np.stack(batch.state), np.stack(batch.action))
    with torch.no_grad():
        values = value_net(states)
        values_exp = value_net_exp(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    advantages_exp, returns_exp = estimate_advantages(rewards_exp, masks, values_exp, args.gamma, args.tau, device)

    """update discriminator"""
    for _ in range(1):
        expert_state_actions = torch.from_numpy(partial_expert_traj).to(dtype).to(device)
        partial_states = states[:, :args.observe]
        g_o = discrim_net(torch.cat([partial_states, actions], 1))
        e_o = discrim_net(expert_state_actions)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
                       discrim_criterion(e_o, zeros((partial_expert_traj.shape[0], 1), device=device))
        discrim_loss.backward()
        optimizer_discrim.step()

    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)
    if (kl > 6e-7):
        trpo_step(policy_net, value_net_exp, states, actions, returns_exp, advantages_exp, kl, args.damping,
                  args.l2_reg, fixed_log_probs=fixed_log_probs)


def main_loop(agent):
    kl = args.delta_0
    prev_rwd = deque(maxlen=args.window)
    prev_rwd.append(0)

    model_path = create_model_path(args)
    s_name = '/model_mu.pt' if args.env_name == 'mujoco' else '/model_py.pt'
    csv_name = '/train_info_mu.csv' if args.env_name == 'mujoco' else '/train_info_py.csv'
    train_info = {'steps': [], 'rewards': [], 'times': []}
    print(f'epochs: {args.max_iter_num}, episodes: {args.max_iter_num*args.min_batch_size}, steps: {args.max_iter_num*args.min_batch_size*args.episode_length}')
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size, ep_len=args.episode_length, render=render)
        discrim_net.to(device)

        if (args.K_delta > -1):
            if (i_iter > args.K_delta):
                avg_prev_rwd = np.mean(prev_rwd)
                if (avg_prev_rwd < log['avg_reward']):
                    kl = max(args.low_kl, kl * args.delta)
        # writer.add_scalar('KL', kl, i_iter + 1)
        prev_rwd.append(log['avg_reward'])
        t0 = time.time()
        update_params(batch, i_iter, kl)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""
        discrim_net.to(torch.device('cpu'))
        _, log_eval = agent.collect_samples(args.eval_batch_size, ep_len=args.episode_length, eval_flag=True, mean_action=True, render=render)
        discrim_net.to(device)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            print('[{}]\tepoch: {}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}\t KL{:}'
                  .format(datetime.now(), i_iter,  log['avg_reward'], log_eval['avg_reward'], kl))
            train_info['steps'].append(i_iter*args.min_batch_size*args.episode_length)
            train_info['rewards'].append(log_eval['avg_reward'])
            train_info['times'].append(datetime.now())
            par.save_data_to_csv(train_info['steps'], train_info['rewards'], model_path+csv_name, time=train_info['times'])
            agent.save(model_path+s_name)

        # writer.add_scalar('rewards/train_R_avg', log['avg_reward'], i_iter + 1)
        # writer.add_scalar('rewards/eval_R_avg', log_eval['avg_reward'], i_iter + 1)

        """clean up gpu memory"""
        torch.cuda.empty_cache()
