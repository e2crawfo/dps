import numpy as np
import time
import datetime

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.updater import DifferentiableUpdater
from dps.envs.grid_arithmetic import GridArithmeticDataset
from dps.environment import RegressionEnv
from dps.utils.tf import LeNet
from dps.utils import Config


def build_env():
    train = GridArithmeticDataset(n_examples=cfg.n_train)
    val = GridArithmeticDataset(n_examples=cfg.n_val)
    test = GridArithmeticDataset(n_examples=cfg.n_val)
    return RegressionEnv(train, val, test)


def get_updater(env):
    build_model = LeNet(n_units=int(cfg.n_controller_units))
    return DifferentiableUpdater(env, build_model)


config = DEFAULT_CONFIG.copy(
    name="CNNExperiment",

    n_val=100,

    batch_size=64,
    n_controller_units=256,
    log_name="cnn_grid_arithmetic",
    max_steps=1000000,
    display_step=100,
    eval_step=100,
    patience=5000,
    reward_window=0.499,
    one_hot_output=True,
    loss_type="xent",
    stopping_function=lambda val_record: -val_record['reward'],
    preserve_policy=True,

    slim=False,
    save_summaries=True,
    start_tensorboard=True,
    verbose=False,
    show_plots=True,
    save_plots=True,

    use_gpu=True,
    gpu_allow_growth=True,
    threshold=0.01,
    memory_limit_mb=12*1024
)

env_config = Config(
    build_env=build_env,
    shape=(2, 2),
    mnist=True,
    min_digits=2,
    max_digits=3,
    reductions="sum",
    op_loc=None,
    base=10,
    largest_digit=100,
    final_reward=True,
)

alg_config = Config(
    get_updater=get_updater,
    optimizer_spec="adam",
    # curriculum=[dict(lr_schedule=lrs) for lrs in [1e-3, 1e-4, 1e-5, 1e-6]],
    lr_schedule=1e-4,
    power_through=True,
    noise_schedule=0.0,
    max_grad_norm=None,
    l2_weight=0.0,
)

config.update(alg_config)
config.update(env_config)

# grid = dict(n_train=2**np.array([6, 17]), n_controller_units=2**np.array([5, 10]))

grid = dict(n_train=2**np.arange(6, 18), n_controller_units=2**np.arange(5, 11))

from dps.parallel.hyper import build_and_submit
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=host_pool)
