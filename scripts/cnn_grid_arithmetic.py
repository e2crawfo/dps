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
from dps.vision import LeNet


def build_env():
    train = GridArithmeticDataset(n_examples=cfg.n_train)
    val = GridArithmeticDataset(n_examples=cfg.n_val)
    return RegressionEnv(train, val)


config = DEFAULT_CONFIG.copy(
    display_step=10,

    n_train=10000,
    n_val=32,
    shape=(2, 2),
    start_loc=(0, 0),
    mnist=True,
    min_digits=2,
    max_digits=3,
    ablation='',
    dense_reward=False,
    reductions={
        'A': lambda x: sum(x),
        'M': lambda x: np.product(x),
        'C': lambda x: len(x),
        'X': lambda x: max(x),
        'N': lambda x: min(x)
    },
    op_loc=None,
    # op_loc=(0, 0),
    base=10,
    n_controller_units=256,
    build_env=build_env,
    log_name="cnn_grid_arithmetic",
    max_steps=100000,

    optimizer_spec="adam",
    lr_schedule="1e-4",
    noise_schedule=0.0,
    max_grad_norm=None,

    threshold=0.0,

    reward_window=0.1,

    one_hot_output=True,
    largest_digit=30,

    loss_type="xent",
    downsample_factor=2,

    stopping_function=lambda val_record: -val_record['reward']
)


def get_updater(env):
    output_size = cfg.largest_digit + 2 if cfg.loss_type == "xent" else 1
    return DifferentiableUpdater(env, LeNet(n_units=cfg.n_controller_units, output_size=output_size))


config.get_updater = get_updater


start_time = time.time()
print("Starting new training run at: ")
print(datetime.datetime.now())

with config:
    cl_args = clify.wrap_object(cfg).parse()
    config.update(cl_args)

    val = training_loop(start_time=start_time)
