import numpy as np
import time
import datetime

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.updater import DifferentiableUpdater
from dps.experiments.alt_arithmetic import AltArithmeticEnv
# from dps.utils import MLP
from dps.vision import LeNet


def build_env():
    return AltArithmeticEnv(
        mnist=True, shape=cfg.shape, n_digits=cfg.n_digits,
        upper_bound=cfg.upper_bound, base=cfg.base, n_train=cfg.n_train,
        n_val=cfg.n_val, op_loc=cfg.op_loc,
        start_loc=cfg.start_loc, force_2d=cfg.force_2d)


config = DEFAULT_CONFIG.copy(
    force_2d=True,
    n_train=10000,
    n_val=1000,
    shape=(2, 2),
    start_loc=(0, 0),
    n_digits=3,
    symbols=[
        ('A', lambda x: sum(x)),
        ('M', lambda x: np.product(x)),
        ('C', lambda x: len(x))],
    op_loc=(0, 0),
    upper_bound=True,
    base=10,
    n_controller_units=256,
    get_updater=lambda env: DifferentiableUpdater(env, LeNet(n_units=cfg.n_controller_units, output_size=1)),
    # get_updater=lambda env: DifferentiableUpdater(env, MLP([cfg.n_controller_units, cfg.n_controller_units])),
    build_env=build_env,
    log_name="cnn_alt_arithmetic",
    max_steps=100000,
)

start_time = time.time()
print("Starting new training run at: ")
print(datetime.datetime.now())

with config:
    cl_args = clify.wrap_object(cfg).parse()
    config.update(cl_args)

    val = training_loop(start_time=start_time)
