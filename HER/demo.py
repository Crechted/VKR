import torch
from HER.rl_modules.models import actor
import numpy as np
import configs.parse_csv as parse
import os
import time

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
    s_name = '/model_mu.pt' if args.env_prog == 'mujoco' else '/model_py.pt'
    model_path = cwd + '/HER/saved_models/' + args.env_name + s_name
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    st, rew = [], []
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        rewards = 0
        for t in range(env._max_episode_steps):
            if args.env_prog == 'mujoco' and args.render:
                env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            rewards += reward
            time.sleep(0.02)
        print(f'the episode is: {i}, reward: {rewards}')
        rew.append(rewards)
        st.append(i)

        parse.save_data_to_csv(st, rew, 'test.csv')


