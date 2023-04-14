import numpy as np

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
    if cfg.env_name == 'reach':
        cfg.action_repeat = 1
        cfg.epochs = 14
        cfg.n_episodes = 3
    elif cfg.env_name == 'slide':
        cfg.action_repeat = 1
        cfg.epochs = 200
        cfg.n_episodes = 18
    elif cfg.env_name == 'push':
        cfg.action_repeat = 1
        cfg.epochs = 15
        cfg.n_episodes = 18
    elif cfg.env_name == 'pick_and_place':
        cfg.action_repeat = 1
        cfg.epochs = 50
        cfg.n_episodes = 18


    cfg.discount = 0.99
    cfg.episode_length = env._max_episode_steps

    # planning
    # cfg.iterations = 6
    cfg.num_samples = 256
    # cfg.num_elites = 64
    # cfg.mixture_coef = 0.05
    # cfg.min_std = 0.05
    # cfg.temperature = 0.5
    # cfg.momentum = 0.1

    # learning
    cfg.batch_size = 512
    # cfg.max_buffer_size = 1000000
    # cfg.horizon = 5
    # cfg.reward_coef = 0.5
    # cfg.value_coef = 0.1
    # cfg.consistency_coef = 2
    # cfg.rho = 0.5
    # cfg.kappa = 0.1
    # cfg.lr = 1e-3

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
    cfg.exp_name = args.env_name
    cfg.eval_freq = cfg.n_cycles*cfg.n_episodes*cfg.episode_length
    cfg.eval_episodes = 10
    cfg.train_steps = int(cfg.eval_freq*cfg.epochs)

    cfg.save_video = False
    cfg.save_model = False
    cfg.obs_shape = tuple(x+y for x in env.observation_space['observation' if args.env_prog == 'mujoco' else 'state'].shape for y in env.observation_space['desired_goal'].shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    cfg.device = 'cuda' if args.cuda else 'cpu'
    cfg.save_dir = 'TD_MPC/saved_models/'
    cfg.modality = 'state'
    if args.img_obs:
        cfg.modality = 'pixels'
        cfg.obs_shape = tuple(int(x) for x in env.observation_space['observation' if args.env_prog == 'mujoco' else 'state'].shape)
        cfg.frame_stack = 224/3
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
    cfg.tau = 0.01
    if (cfg.env_name == 'reach'):
        cfg.demo_traj_path = 'LOGO/logo/data/HalfCheetah-v2_data.p'
        cfg.K_delta = 50
        cfg.sparse_val = 2.
        cfg.delta_0 = 0.2
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


# FetchReach-v1 = {'obs': 10, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
# FetchSlide-v1 = {'obs': 25, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
# FetchPush-v1 = {'obs': 25, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
# FetchPickAndPlace-v1 = {'obs': 25, 'goal': 3, 'action': 4, 'action_max': 1.0, 'max_timesteps': 50}
