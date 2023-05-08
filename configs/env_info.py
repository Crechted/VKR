import numpy as np
import os


def get_HER_cfgs(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = env._max_episode_steps
    return params


def get_TDMPC_cfgs(args, env):
    cfg = args

    cfg.n_cycles = 50
    if cfg.task_name == 'reach':
        cfg.action_repeat = 1
        cfg.n_epochs = 14
        cfg.n_episodes = 3
    elif cfg.task_name == 'slide':
        cfg.action_repeat = 1
        cfg.n_epochs = 200
        cfg.n_episodes = 18
    elif cfg.task_name == 'push':
        cfg.action_repeat = 1
        cfg.n_epochs = 8
        cfg.n_episodes = 18
    elif cfg.task_name == 'pick_and_place':
        cfg.action_repeat = 1
        cfg.n_epochs = 40
        cfg.n_episodes = 18

    cfg.discount = 0.99
    cfg.episode_length = env._max_episode_steps

    # planning
    # cfg.iterations = 6
    # cfg.num_samples = 256
    # cfg.num_elites = 64
    # cfg.mixture_coef = 0.05
    # cfg.min_std = 0.05
    # cfg.temperature = 0.5
    # cfg.momentum = 0.1

    # learning
    # cfg.batch_size = 512
    # cfg.max_buffer_size = 1000000
    # cfg.horizon = 5
    # cfg.reward_coef = 0.5
    # cfg.value_coef = 0.1
    # cfg.consistency_coef = 2
    # cfg.rho = 0.5
    # cfg.kappa = 0.1
    # cfg.lr = 1e_3

    cfg.std_schedule = f'linear(0.5, {cfg.min_std}, 25000)'
    cfg.horizon_schedule = f'linear(1, {cfg.horizon}, 25000)'
    # cfg.per_alpha = 0.6
    # cfg.per_beta = 0.4
    # cfg.grad_clip_norm = 10
    # cfg.seed_steps = 250
    # cfg.update_freq = 2
    # cfg.tau = 0.01

    # architecture
    # cfg.enc_dim = 256
    # cfg.mlp_dim = 512
    # cfg.latent_dim = 50

    # wandb (insert your own)
    cfg.use_wandb = False

    # misc
    cfg.seed = 22
    cfg.exp_name = args.task_name
    cfg.eval_freq = cfg.n_cycles * cfg.n_episodes * cfg.episode_length
    cfg.eval_episodes = 10
    cfg.train_steps = int(cfg.eval_freq * cfg.n_epochs)

    cfg.save_video = False
    cfg.save_model = False
    cfg.obs_shape = tuple(
        x + y for x in env.observation_space['observation' if args.env_name == 'mujoco' else 'state'].shape for y in
        env.observation_space['desired_goal'].shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    cfg.device = 'cuda' if args.cuda else 'cpu'
    cfg.save_dir = 'TD_MPC/saved_models/'
    cfg.modality = 'state'
    if args.img_obs:
        cfg.modality = 'pixels'
        cfg.obs_shape = tuple(
            int(x) for x in env.observation_space['observation' if args.env_name == 'mujoco' else 'state'].shape)
        cfg.frame_stack = 224 / 3
        cfg.num_channels = 32
        cfg.img_size = 224
        cfg.lr = 3e-4
        cfg.batch_size = 256
    return cfg


def get_LOGO_cfgs(args, env):
    cfg = args
    cfg.save_dir = 'LOGO/saved_models/'
    cfg.episode_length = env._max_episode_steps
    cfg.gamma = 0.99
    cfg.tau = 0.5
    if (cfg.task_name == 'reach'):
        cfg.demo_traj_path = 'LOGO/logo/data/HalfCheetah_v2_data.p'
        cfg.K_delta = 50
        cfg.sparse_val = 2.
        # cfg.delta_0 = 0.2
        cfg.low_kl = 5e-7
        cfg.delta = 0.95
        cfg.min_batch_size = 1000
        cfg.seed = 11
        cfg.observe = 0
        cfg.max_iter_num = 2000
        if cfg.init_BC:
            cfg.delta_0 = 0.05
        cfg.delay_val = 1000
    elif (cfg.task_name == 'push'):
        cfg.demo_traj_path = 'LOGO/logo/data/HalfCheetah_v2_data.p'
        cfg.K_delta = 50
        cfg.sparse_val = 2.
        # cfg.delta_0 = 0.2
        cfg.low_kl = 5e-7
        cfg.delta = 0.95
        cfg.min_batch_size = 1000
        cfg.seed = 11
        cfg.observe = 0
        cfg.max_iter_num = 2000
        if cfg.init_BC:
            cfg.delta_0 = 0.05
        cfg.delay_val = 1000
    elif (cfg.task_name == 'pick_and_place'):
        cfg.demo_traj_path = 'LOGO/logo/data/HalfCheetah_v2_data.p'
        cfg.K_delta = 50
        cfg.sparse_val = 2.
        # cfg.delta_0 = 0.2
        cfg.low_kl = 5e-7
        cfg.delta = 0.95
        cfg.min_batch_size = 1000
        cfg.seed = 11
        cfg.observe = 0
        cfg.max_iter_num = 2000
        if cfg.init_BC:
            cfg.delta_0 = 0.05
        cfg.delay_val = 1000
    elif (cfg.task_name == 'slide'):
        cfg.demo_traj_path = 'LOGO/logo/data/HalfCheetah_v2_data.p'
        cfg.K_delta = 50
        cfg.sparse_val = 2.
        # cfg.delta_0 = 0.2
        cfg.low_kl = 5e-7
        cfg.delta = 0.95
        cfg.min_batch_size = 1000
        cfg.seed = 11
        cfg.observe = 0
        cfg.max_iter_num = 2000
        if cfg.init_BC:
            cfg.delta_0 = 0.05
        cfg.delay_val = 1000
    return cfg


def get_cfgs(args, env):
    if args.alg_name == 'HER':
        return get_HER_cfgs(env)
    elif args.alg_name == 'TD_MPC':
        return get_TDMPC_cfgs(args, env)
    elif args.alg_name == 'LOGO':
        return get_LOGO_cfgs(args, env)

def create_model_path(cfgs):
	if not os.path.exists(cfgs.save_dir):
		os.mkdir(cfgs.save_dir)
	# path to save the model
	model_path = os.path.join(cfgs.save_dir, cfgs.task_name)
	if not os.path.exists(model_path):
		os.mkdir(model_path)
	return model_path
class Configs(object):
    def __init__(self, cfg):
        self.alg_name = cfg.alg_name
        self.task_name = cfg.task_name
        self.env_name = cfg.env_name
        self.seed = cfg.seed
        self.n_epochs = cfg.n_epochs
        self.n_cycles = cfg.n_cycles
        self.n_batches = cfg.n_batches
        if cfg.alg_name == 'HER':
            self.save_interval = cfg.save_interval
            self.num_workers = cfg.num_workers
            self.replay_strategy = cfg.replay_strategy
            self.clip_return = cfg.clip_return
            self.noise_eps = cfg.noise_eps
            self.random_eps = cfg.random_eps
            self.buffer_size = cfg.buffer_size
            self.replay_k = cfg.replay_k
            self.clip_obs = cfg.clip_obs
            self.batch_size = cfg.batch_size
            self.gamma = cfg.gamma
            self.action_l2 = cfg.action_l2
            self.lr_actor = cfg.lr_actor
            self.lr_critic = cfg.lr_critic
            self.polyak = cfg.polyak
            self.n_test_rollouts = cfg.n_test_rollouts
            self.clip_range = cfg.clip_range
            self.demo_length = cfg.demo_length
            self.num_rollouts_per_mpi = cfg.num_rollouts_per_mpi
        elif cfg.alg_name == 'TD_MPC':
            self.iterations = cfg.iterations
            self.num_samples = cfg.num_samples
            self.num_elites = cfg.num_elites
            self.mixture_coef = cfg.mixture_coef
            self.min_std = cfg.min_std
            self.temperature = cfg.temperature
            self.momentum = cfg.momentum
            self.max_buffer_size = cfg.max_buffer_size
            self.horizon = cfg.horizon
            self.reward_coef = cfg.reward_coef
            self.value_coef = cfg.value_coef
            self.consistency_coef = cfg.consistency_coef
            self.rho = cfg.rho
            self.kappa = cfg.kappa
            self.lr = cfg.lr
            self.std_schedule = cfg.std_schedule
            self.horizon_schedule = cfg.horizon_schedule
            self.per_alpha = cfg.per_alpha
            self.per_beta = cfg.per_beta
            self.grad_clip_norm = cfg.grad_clip_norm
            self.seed_steps = cfg.seed_steps
            self.update_freq = cfg.update_freq
            self.tau = cfg.tau
            self.enc_dim = cfg.enc_dim
            self.mlp_dim = cfg.mlp_dim
            self.latent_dim = cfg.latent_dim            
            self.batch_size = cfg.batch_size
            self.discount = cfg.discount
            self.eval_freq = cfg.eval_freq
            self.eval_episodes = cfg.eval_episodes
            self.train_steps = cfg.train_steps
        elif cfg.alg_name == 'LOGO':
            self.gamma = cfg.gamma
            self.max_kl = cfg.max_kl
            self.damping = cfg.damping
            self.log_std = cfg.log_std
            self.l2_reg = cfg.l2_reg
            self.learning_rate = cfg.learning_rate
            self.clip_epsilon = cfg.clip_epsilon
            self.num_threads = cfg.num_threads
            self.min_batch_size = cfg.min_batch_size
            self.eval_batch_size = cfg.eval_batch_size
            self.max_iter_num = cfg.max_iter_num
            self.log_interval = cfg.log_interval
            self.gpu_index = cfg.gpu_index
            self.init_BC = cfg.init_BC
            self.env_num = cfg.env_num
            self.window = cfg.window
            self.nn_param = cfg.nn_param
            self.K_delta = cfg.K_delta
            self.delta_0 = cfg.delta_0
            self.delta = cfg.delta
            self.tau = cfg.tau
            self.sparse_val = cfg.sparse_val
            self.low_kl = cfg.low_kl
            self.observe = cfg.observe
            self.delay_val = cfg.delay_val
        self.save_dir = cfg.save_dir

    def demo(self):
        print(str(self.__dict__))
        model_path = create_model_path(self)
        s_name = '/settings_mu.txt' if self.env_name == 'mujoco' else '/settings_py.txt'
        with open(model_path+s_name, 'w') as f:
            f.write(str(self.__dict__))

# FetchReach_v1 = {'obs': 10, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
# FetchSlide_v1 = {'obs': 25, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
# FetchPush_v1 = {'obs': 25, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
# FetchPickAndPlace_v1 = {'obs': 25, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
