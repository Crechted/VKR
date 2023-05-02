import HER
from HER import demo, train
import TD_MPC
from TD_MPC import demo, train
import LOGO
from LOGO import demo, train
import pybullet_multigoal_gym as pmg
import configs.arguments as get_arg
import gym

# set PATH=C:\Users\Crechted\.mujoco\mjpro150\bin;%PATH%
# os.add_dll_directory("C://Users//Crechted//.mujoco//mjpro150//bin")
# mpirun -np 1 python main.py --alg-name='HER' --task-name='reach' | tee HER/logs/reach.log
# python main.py --alg-name="TD_MPC" --task-name="reach" --cuda --render | tee TD_MPC/logs/reach.log
# python main.py --alg-name='LOGO' --task-name='reach' --render | tee LOGO/logs/reach.log

# python main.py --alg-name='TD_MPC' --task-name='push' --env-name='mujoco' --cuda --load --horizon=7
# numba 0.56.4 requires numpy<1.24,>=1.18, but you have numpy 1.24.2 which is incompatible.

# python main.py --alg-name="HER" --task-name="reach" --env-name="mujoco" --demo --render
# python main.py --alg-name="TD_MPC" --task-name="push" --seed-steps=250 --horizon=2 --iterations=3 --env-name="mujoco" --cuda --load
def create_env(args):
    if args.env_name == "mujoco":
        return create_mujoco_env(args)
    elif args.env_name == "pybullet":
        return create_pybullet_env(args)


def create_mujoco_env(args):
    if args.task_name == 'reach':
        return gym.make('FetchReach-v1')
    elif args.task_name == 'slide':
        return gym.make('FetchSlide-v1')
    elif args.task_name == 'push':
        return gym.make('FetchPush-v1')
    elif args.task_name == 'pick_and_place':
        return gym.make('FetchPickAndPlace-v1')
    print('bruh')


def create_pybullet_env(args):
    camera_setup = [
        {
            'cameraEyePosition': [0.0, -0.0, 0.4],
            'cameraTargetPosition': [-0.45, -0.0, 0.0],
            'cameraUpVector': [0, 0, 1],
            'render_width': 224,
            'render_height': 224
        },
        {
            'cameraEyePosition': [-1.0, -0.25, 0.6],
            'cameraTargetPosition': [-0.6, -0.05, 0.2],
            'cameraUpVector': [0, 0, 1],
            'render_width': 224,
            'render_height': 224
        },
    ]

    return pmg.make_env(
        # task args ['reach', 'push', 'slide', 'pick_and_place',
        #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
        task=args.env_name,
        gripper='parallel_jaw',
        # num_block=2,  # only meaningful for multi-block tasks, up to 5 blocks
        render=args.render,
        joint_control=False,
        binary_reward=args.binary_rew,
        # image observation args
        camera_setup=camera_setup,
        image_observation=args.img_obs,
        observation_cam_id=[0],
        goal_cam_id=1)


def choose_learn_alg(a):
    if a.alg_name == 'HER':
        return HER
    if a.alg_name == 'TD_MPC':
        return TD_MPC
    if a.alg_name == 'LOGO':
        return LOGO
    print('bruh')


if __name__ == '__main__':
    args, parser = get_arg.get_default_args_and_parser()
    args = get_arg.get_args_by_alg(args.alg_name, parser)

    print("RENDER: ", args.render)
    env = create_env(args)
    alg = choose_learn_alg(args)
    print(args)
    if args.demo:
        alg.demo.launch(args, env)
    else:
        alg.train.launch(args, env)
