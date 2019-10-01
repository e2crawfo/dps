from dps.utils import Config
from dps.iris_example import iris_config, mlp_config
from dps.hyper import run_experiment

env_configs = {'iris': iris_config}
alg_configs = {'mlp': mlp_config}


if __name__ == "__main__":
    config = Config()
    run_experiment(
        "iris_mlp_experiment", config, "", cl_mode='strict',
        env_configs=env_configs, alg_configs=alg_configs)
