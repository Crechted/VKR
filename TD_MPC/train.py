import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from TD_MPC.src.algorithm.tdmpc import TDMPC
from TD_MPC.src.algorithm.helper import Episode, ReplayBuffer
from configs.env_info import get_TDMPC_cfgs
from datetime import datetime
import configs.parse_csv as parse
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def create_model_path(cfgs):
	if not os.path.exists(cfgs.save_dir):
		os.mkdir(cfgs.save_dir)
	# path to save the model
	model_path = os.path.join(cfgs.save_dir, cfgs.env_name)
	if not os.path.exists(model_path):
		os.mkdir(model_path)
	return model_path


def get_obs(obs):
	g = np.append(obs['observation'], obs['desired_goal'])
	return g


def evaluate(cfg, env, agent, num_episodes, step):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			if cfg.env_prog == 'mujoco' and cfg.render:
				env.render()
			action = agent.plan(get_obs(obs), eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			t += 1
		episode_rewards.append(ep_reward)
	return np.nanmean(episode_rewards)


def launch(args, env):
	"""Training script for TD_MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	cfg = get_TDMPC_cfgs(args, env)
	set_seed(cfg.seed)
	model_path = create_model_path(cfg)
	s_name = '/model_mu.pt' if cfg.env_prog == 'mujoco' else '/model_py.pt'
	csv_name = '/train_info_mu.csv' if cfg.env_prog == 'mujoco' else '/train_info_py.csv'
	agent, buffer = TDMPC(cfg), ReplayBuffer(cfg)

	# Run training
	# L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()
	train_info = {'steps': [], 'rewards': []}
	if cfg.load:
		print(f'load {model_path+s_name}')
		agent.load(model_path+s_name)
		train_info['steps'], train_info['rewards'] = parse.load_data_from_csv(model_path+csv_name)
		print(f'was loaded: {train_info}')

	print(f"\nNUM STEPS:{cfg.train_steps}, num epochs: {cfg.epochs}\n OBS:{env.observation_space}")
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, get_obs(obs))
		while not episode.done:
			if cfg.env_prog == 'mujoco' and cfg.render:
				env.render()
			action = agent.plan(get_obs(obs), step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (get_obs(obs), action, reward, done)
		assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}

		train_metrics.update(common_metrics)
		# L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			print(f'\n[{datetime.now()}] evaluate № {int(env_step/cfg.eval_freq)}')
			common_metrics['episode_reward'] = evaluate(cfg, env, agent, cfg.eval_episodes, step)
			print(common_metrics)
			train_info['steps'].append(common_metrics['step'])
			train_info['rewards'].append(common_metrics['episode_reward'])
			parse.save_data_to_csv(train_info['steps'], train_info['rewards'], model_path + csv_name)
			agent.save(model_path + s_name)
			# L.log(common_metrics, category='eval')

	# L.finish(agent)
	print('Training completed successfully')
