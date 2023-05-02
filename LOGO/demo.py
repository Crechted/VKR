from LOGO.models.mlp_policy import Policy
from LOGO.models.mlp_policy_disc import DiscretePolicy
from LOGO.utils.torch import *
import os
from configs.env_info import get_cfgs

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs
def get_obs(obs):
    g = np.append(obs['observation'], obs['desired_goal'])
    return g
def launch(args, env):
    # load the model param
    cfg = get_cfgs(args, env)
    cwd = os.getcwd()
    s_name = '/model_mu.pt' if args.env_name == 'mujoco' else '/model_py.pt'
    model_path = cwd + '/LOGO/saved_models/' + args.task_name + s_name

    nn_size = tuple(args.nn_param)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    obs_shape = tuple(
        x + y for x in env.observation_space['observation' if args.env_name == 'mujoco' else 'state'].shape for y in env.observation_space['desired_goal'].shape)
    state_dim = obs_shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std, hidden_size=nn_size)

    policy_net.load_state_dict(torch.load(model_path))

    for i in range(args.demo_length):
        state = get_obs(env.reset())
        rewards = 0
        for t in range(env._max_episode_steps):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy_net(state_var)[0][0].numpy()
            action = int(action) if policy_net.is_disc_action else action.astype(np.float64)
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            state = get_obs(obs)
            if args.render and args.env_name == 'mujoco':
                env.render()
        print(f'the episode is: {i}, reward: {rewards}')
    # get the environment params
    # create the actor network
    # print(f'the episode is: {i}, reward: {rewards}')
