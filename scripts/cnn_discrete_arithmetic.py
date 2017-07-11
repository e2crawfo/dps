import numpy as np
import time
import datetime

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.utils import FeedforwardCell
from dps.updater import DifferentiableUpdater
from dps.experiments.alt_arithmetic import AltArithmeticEnv
from dps.policy import Deterministic
from dps.mnist import LeNet


def build_env():
    return AltArithmeticEnv(
        mnist=True, shape=cfg.shape, n_digits=cfg.n_digits,
        upper_bound=cfg.upper_bound, base=cfg.base, n_train=cfg.n_train,
        n_val=cfg.n_val, n_test=cfg.n_test, op_loc=cfg.op_loc,
        start_loc=cfg.start_loc, force_2d=cfg.force_2d)


config = DEFAULT_CONFIG.copy(
    force_2d=True,
    n_train=10000,
    n_val=1000,
    n_test=0,
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
    build_updater=DifferentiableUpdater,
    build_env=build_env,
    n_controller_units=1000,
    action_selection=lambda env: Deterministic(env.n_actions),
    controller=lambda n_params: FeedforwardCell(LeNet(n_units=cfg.n_controller_units), n_params),
    log_name="cnn_alt_arithmetic"
)

start_time = time.time()
print("Starting new training run at: ")
print(datetime.datetime.now())

with config:
    cl_args = clify.wrap_object(cfg).parse()
    config.update(cl_args)

    val = training_loop(start_time=start_time)
