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
from dps.vision import LeNet, VGGNet
from dps.utils import Config


def build_env():
    train = GridArithmeticDataset(n_examples=cfg.n_train)
    val = GridArithmeticDataset(n_examples=cfg.n_val)
    return RegressionEnv(train, val)


def get_updater(env):
    # output_size = cfg.largest_digit + 2 if cfg.loss_type == "xent" else 1
    build_model = LeNet(n_units=int(cfg.n_controller_units))
    # build_model = LeNet(n_units=cfg.n_controller_units)
    return DifferentiableUpdater(env, build_model)


do_search = True


config = DEFAULT_CONFIG.copy(
    name="CNNExperiment",

    n_val=1000,

    batch_size=64,
    n_controller_units=256 if not do_search else None,
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
    preserve_env=True,
    slim=do_search,
    save_summaries=not do_search,
    start_tensorboard=not do_search,
    verbose=False,
    visualize=True,
    display=False,
    save_display=False,
    use_gpu=True,
    threshold=0.05,
    memory_limit_mb=12*1024
)

env_config = Config(
    build_env=build_env,
    shape=(2, 2),
    mnist=True,
    min_digits=2,
    max_digits=3,
    reductions=lambda x: sum(x),
    #     lambda x: np.product(x),
    #     lambda x: len(x),
    #     lambda x: max(x),
    #     lambda x: min(x)
    op_loc=None,
    base=10,
    largest_digit=26,
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


if do_search:
    from dps.parallel.hyper import build_and_submit
    grid = dict(n_train=2**np.arange(10, 18), n_controller_units=2**np.arange(5, 11))
    host_pool = [':']
    clify.wrap_function(build_and_submit)(config, grid, n_param_settings=None, host_pool=host_pool)
else:
    start_time = time.time()
    print("Starting new training run at: ")
    print(datetime.datetime.now())

    with config:
        cl_args = clify.wrap_object(cfg).parse()
        config.update(cl_args)

        val = training_loop(start_time=start_time)
