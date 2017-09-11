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
    max_steps=20000,
    display_step=100,
    eval_step=10,
    patience=np.inf,
    power_through=False,
    preserve_policy=True,
    slim=True,
    save_summaries=False,
    start_tensorboard=False,
    verbose=False,
    visualize=False,
    display=False,
    save_display=False,
    use_gpu=False,
    reward_window=0.1,
    threshold=0.05,
)


alg_config = Config(
    get_updater=ppo.PPO,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",

    exploration_schedule=(
        "MixtureSchedule("
        "    [Poly(10, {0}, end=5.0), Poly(10, {0}, end=1.0), Poly(10, {0}, end=0.1)],"
        "    100, shared_clock=False)"
    ).format(config.max_steps),
    test_time_explore=0.1,

    policy_weight=1.0,
    lr_schedule=1e-4,
    n_controller_units=128,
    batch_size=16,
    value_weight=1.0,
    entropy_weight=0.01,
    gamma=1.0,
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

    dense_reward=True,
    reward_window=0.499,

    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.
    log_name='grid_arithmetic',
    render_rollouts=None
)


config.update(alg_config)
config.update(env_config)


grid = dict(
    # opt_steps_per_update=np.linspace(1, 21, 11).astype('i'),
    # epsilon=list(np.linspace(0.04, 0.4, 10)) + [None],
    opt_steps_per_update=[1, 2, 3, 4],
    epsilon=[0.2, None],
)


from dps.parallel.hyper import build_and_submit
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(17, 33)]
clify.wrap_function(build_and_submit)(
    config, grid, max_hosts=4, ppn=2, n_repeats=4, walltime="00:15:00",
    cleanup_time="00:03:00", host_pool=host_pool
)
