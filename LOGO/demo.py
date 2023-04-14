import torch
import numpy as np
from LOGO.models.mlp_policy import Policy
from LOGO.models.mlp_policy_disc import DiscretePolicy
import os

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
    # load the model param

    cwd = os.getcwd()
    model_path = cwd + '/LOGO/saved_models/' + args.env_name + '/model_mu.pt'

    nn_size = tuple(args.nn_param)
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std, hidden_size=nn_size)

    policy_net.load_state_dict(torch.load(args.model_path))

    obs = env.reset()
    # get the environment params
    # create the actor network
    # print(f'the episode is: {i}, reward: {rewards}')
