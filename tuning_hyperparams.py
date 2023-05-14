from ConfigSpace import Configuration, ConfigurationSpace
from LOGO import train as train_logo
from HER import train as train_her
from smac import HyperparameterOptimizationFacade, Scenario
import os

def create_model_path(cfgs):
    if not os.path.exists(cfgs.save_dir):
        os.mkdir(cfgs.save_dir)
    # path to save the model
    model_path = os.path.join(cfgs.save_dir, cfgs.task_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    return model_path

class Tune(object):
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

    def train_LOGO(self, config: Configuration, seed: int = 0) -> float:
        self.cfg.damping = config["damping"]
        self.cfg.log_std = config["log-std"]
        self.cfg.l2_reg = config["l2-reg"]
        self.cfg.learning_rate = config["learning-rate"]
        self.cfg.clip_epsilon = config["clip-epsilon"]
        self.cfg.gamma = config["gamma"]
        self.cfg.tau = config["tau"]
        self.cfg.sparse_val = config["sparse-val"]
        self.cfg.K_delta = config["K-delta"]
        self.cfg.delta_0 = config["delta-0"]
        self.cfg.seed = config["seed"]
        print(f"train with {config}")
        eval_rewards = train_logo.launch(self.cfg, self.env)
        scores = (50.0+eval_rewards)/50.0
        print(f"RESULT: {eval_rewards} --- {scores} ")

        return scores

    def train_HER(self, config: Configuration, seed: int = 0) -> float:
        self.cfg.noise_eps = config["noise-eps"]
        self.cfg.random_eps = config["random-eps"]
        self.cfg.replay_k = config["replay-k"]
        self.cfg.clip_obs = config["clip-obs"]
        self.cfg.batch_size = config["batch-size"]
        self.cfg.gamma = config["gamma"]
        self.cfg.action_l2 = config["action-l2"]
        self.cfg.lr_actor = config["lr-actor"]
        self.cfg.lr_critic = config["lr-critic"]
        self.cfg.polyak = config["polyak"]
        self.cfg.seed = config["seed"]
        print(f"train with {config}")
        eval_rate = train_her.launch(self.cfg, self.env)
        print(f"RESULT: {eval_rate}")
        return eval_rate

    def get_config_by_alg(self, name):
        configparser = None
        if name == "LOGO":
            configspace = ConfigurationSpace({
                "damping": (1e-3, 3e-2),
                "log-std": (0.0, 1.0),
                "l2-reg": (1e-4, 3e-3),
                "learning-rate": (1e-4, 3e-3),
                "clip-epsilon": (0.05, 0.8),
                "gamma": (0.8, 0.99),
                "tau": (0.5, 0.99),
                "sparse-val": (1., 50.),
                "K-delta": (1, 50),
                "delta-0": (1e-2, 0.2),
                "seed": (1, 50)
            })
        elif name == "HER":
            configspace = ConfigurationSpace({
                "noise-eps": (0.01, 1.0),
                "random-eps": (0.01, 1.0),
                "replay-k": (1, 20),
                "clip-obs": (100, 400),
                "batch-size": (128, 512),
                "gamma": (0.8, 0.99),
                "action-l2": (0.5, 3.0),
                "lr-actor": (1e-4, 3e-3),
                "lr-critic": (1e-4, 3e-3),
                "polyak": (0.8, 0.99),
                "seed": (1, 50)
            })
        return configspace

    def get_train_func(self, name):
        if name == "LOGO":
            return self.train_LOGO
        if name == "HER":
            return self.train_HER


    def tuning(self):
        cs = self.get_config_by_alg(self.cfg.alg_name)

        # Scenario object specifying the optimization environment
        scenario = Scenario(cs, name=self.cfg.alg_name + "_" + self.cfg.task_name, deterministic=True, n_trials=self.cfg.n_trials)

        # Use SMAC to find the best configuration/hyperparameters
        train_func = self.get_train_func(self.cfg.alg_name)
        smac = HyperparameterOptimizationFacade(scenario, train_func)
        incumbent = smac.optimize()
        print(incumbent)
        model_path = create_model_path(self.cfg)
        s_name = '/incumbent_tune.txt'
        with open(model_path + s_name, 'w') as f:
            f.write(str(incumbent))

