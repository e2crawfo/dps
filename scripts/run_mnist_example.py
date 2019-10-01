from dps.utils import Config
from dps.mnist_example import mnist_config, mlp_config
from dps.hyper import run_experiment

env_configs = {'mnist': mnist_config}
alg_configs = {'mlp': mlp_config}


if __name__ == "__main__":
    config = Config()
    run_experiment(
        "mnist_mlp_experiment", config, "", cl_mode='strict',
        env_configs=env_configs, alg_configs=alg_configs)
