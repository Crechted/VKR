import argparse

"""
Here are the param for the training

"""

def get_default_args_and_parser():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--alg-name', type=str, default='HER', help='the algorithm name')
    parser.add_argument('--task-name', type=str, default='reach', help='the environment name')
    parser.add_argument('--env-name', type=str, default='mujoco', help='the environment name')
    parser.add_argument('--binary-rew', action='store_true', default=False, help='binary reward')
    parser.add_argument('--render', action='store_true', default=False, help='render learning')
    parser.add_argument('--demo', action='store_true', default=False, help='Demonstration')
    parser.add_argument('--img-obs', action='store_true', default=False, help='observation is image')
    parser.add_argument('--load', action='store_true', default=False, help='Need load')
    parser.add_argument('--save-dir', type=str, default='HER/saved_models/', help='the path to save the models')
    parser.add_argument('--seed', type=int, default=11, metavar='N', help='random seed')

    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    # parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size') # And TD_MPC
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor') # And LOGO
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=3, help='the rollouts per mpi')

    # TD-MPC

    # planning
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--num-elites', type=int, default=64)
    parser.add_argument('--mixture-coef', type=float, default=0.05)
    parser.add_argument('--min-std', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.1)
    # learning
    parser.add_argument('--max-buffer-size', type=int, default=1000000)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--reward-coef', type=float, default=0.5)
    parser.add_argument('--value-coef', type=float, default=0.1)
    parser.add_argument('--consistency-coef', type=int, default=2)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--kappa', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--per-alpha', type=float, default=0.6)
    parser.add_argument('--per-beta', type=float, default=0.4)
    parser.add_argument('--grad-clip-norm', type=int, default=10)
    parser.add_argument('--seed-steps', type=int, default=5000)
    parser.add_argument('--update-freq', type=int, default=2)
    parser.add_argument('--tau', type=int, default=0.01) # And LOGO
    # architecture
    parser.add_argument('--enc-dim', type=int, default=256)
    parser.add_argument('--mlp-dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=50)

    # LOGO
    parser.add_argument('--train-env-name', default=" ", metavar='G', help='Training Env')
    parser.add_argument('--model-path', metavar='G', help='path of pre-trained model')
    parser.add_argument('--demo-traj-path', metavar='G', help='Demonstration trajectories')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G', help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-2, metavar='G', help='damping (default: 1e-2)')
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G', help='gae (default: 3e-4)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N', help='clipping epsilon for PPO')
    parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size for evaluation (default: 2048)')
    parser.add_argument('--max-iter-num', type=int, default=1500, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    parser.add_argument('--init-BC', action='store_true', default=False, help='Initialize with BC policy')
    parser.add_argument('--env-num', type=int, default=-1, metavar='N', help='Env number')
    parser.add_argument('--window', type=int, default=10, metavar='N', help='observation window')
    parser.add_argument('--nn-param', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--K-delta', type=int, default=-1, metavar='N', help='warmup samples befrore decay')
    parser.add_argument('--delta-0', type=float, default=3e-2, metavar='G', help='max kl value (default: 1e-2)')
    parser.add_argument('--delta', type=float, default=0.95, metavar='G', help='KL decay')

    args = parser.parse_args()

    return args, parser

def get_args_by_alg(alg_name, parser):
    args = parser.parse_args()
    return args