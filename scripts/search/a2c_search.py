import numpy as np

from dps.utils import Config
from dps.envs import grid_arithmetic
from dps.rl.algorithms import a2c
from dps.rl.policy import BuildSoftmaxPolicy, BuildLstmController
from dps.config import DEFAULT_CONFIG


config = DEFAULT_CONFIG.copy(
    name="A2CSearch",

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
    get_updater=a2c.A2C,
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
    reward_window=0.4,

    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.
    log_name='grid_arithmetic',
    render_rollouts=None
)


config.update(alg_config)
config.update(env_config)


distributions = dict(
    lr_schedule=['Poly({}, {}, end=1e-6)'.format(i, config.max_steps) for i in [1e-3, 1e-4, 1e-5]],
    n_controller_units=[32, 64, 128],
    split=[True, False],
    batch_size=[8, 16, 32, 64],
    opt_steps_per_update=[1, 2, 4, 8, 16],
    epsilon=[0.0, 0.1, 0.2, 0.3],
    value_weight=[1.0, 2.0, 4.0, 8.0],
    entropy_weight=[0.0] + ['Poly({}, {}, end=0.0)'.format(i, config.max_steps) for i in 0.5**np.arange(1, 5)],
    lmbda=list(np.linspace(0.0, 1.0, 11).astype('f')),
    gamma=list(np.linspace(0.9, 1.0, 11).astype('f')),
)


from dps.parallel.hyper import build_and_submit_search
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(17, 33)]
build_and_submit_search(config, distributions, host_pool=host_pool)
