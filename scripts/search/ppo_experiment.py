import numpy as np

import clify

from dps.utils import Config
from dps.envs import grid_arithmetic
from dps.rl.algorithms import ppo
from dps.rl.policy import BuildSoftmaxPolicy, BuildLstmController
from dps.config import DEFAULT_CONFIG


config = DEFAULT_CONFIG.copy(
    name="PPOExperiment",

    n_train=10000,
    n_val=500,
    max_steps=100000,
    display_step=100,
    eval_step=10,
    patience=np.inf,
    power_through=False,
    preserve_policy=True,
    slim=True,
    save_summaries=False,
    start_tensorboard=False,
    verbose=False,
    display=False,
    save_display=False,
    use_gpu=False,
    threshold=0.05,
)


alg_config = Config(
    get_updater=ppo.PPO,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",

    exploration_schedule=5.0,
    test_time_explore=-1,

    policy_weight=1.0,
    lr_schedule=1e-4,
    n_controller_units=128,
    batch_size=16,
    entropy_weight=0.1,
    gamma=0.98
)


env_config = Config(
    build_env=grid_arithmetic.build_env,
    symbols=[
        ('A', lambda x: sum(x)),
        # ('M', lambda x: np.product(x)),
        # ('C', lambda x: len(x))
    ],

    curriculum=[
        dict(T=40, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    force_2d=False,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.04,
    classification_bonus=0.0,

    updates_per_sample=1,

    reward_window=0.499,

    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.
    log_name='grid_arithmetic',
    render_rollouts=None
)


config.update(alg_config)
config.update(env_config)


grid = dict(
    opt_steps_per_update=np.linspace(1, 51, 11).astype('i'),
    epsilon=list(np.linspace(0.04, 0.4, 10)) + [None],
)


from dps.parallel.hyper import build_and_submit
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
clify.wrap_function(build_and_submit)(config, grid, n_param_settings=None, host_pool=host_pool)
