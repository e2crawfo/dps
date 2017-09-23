import numpy as np

import clify

from dps.utils import Config
from dps.envs import grid_arithmetic
from dps.rl.algorithms import a2c
from dps.rl.policy import BuildEpsilonSoftmaxPolicy, BuildLstmController
from dps.rl import rl_render_hook
from dps.config import DEFAULT_CONFIG


config = DEFAULT_CONFIG.copy(
    name="A2CExperiment",

    n_train=10000,
    n_val=500,
    max_steps=1000000,
    display_step=100,
    eval_step=10,
    patience=np.inf,
    power_through=False,
    preserve_policy=True,
    slim=True,
    save_summaries=False,
    start_tensorboard=False,
    verbose=False,
    visualize=True,
    display=False,
    save_display=False,
    use_gpu=False,
    threshold=0.05,
    render_hook=rl_render_hook,
)


alg_config = Config(
    get_updater=a2c.A2C,
    build_policy=BuildEpsilonSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",

    test_time_explore=0.0,

    policy_weight=1.0,
    value_weight=1.0,
    value_reg_weight=0.0,
    entropy_weight=0.1,

    lr_schedule=1e-4,
    n_controller_units=128,
    batch_size=16,
    gamma=0.98,
    opt_steps_per_update=10,
    actor_exploration_schedule=None,
    epsilon=0.2,
    split=False,

    q_lmbda=1.0,
    v_lmbda=1.0,
    policy_importance_c=0,
    q_importance_c=None,
    v_importance_c=None,
    max_grad_norm=None,

    updates_per_sample=1,
)


env_config = Config(
    build_env=grid_arithmetic.build_env,
    symbols=[
        ('A', lambda x: sum(x)),
    ],
    arithmetic_actions=[
        ('+', lambda acc, digit: acc + digit),
        ('+1', lambda acc, digit: acc + 1),
    ],

    curriculum=[
        dict(T=30, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    force_2d=False,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.04,
    classification_bonus=0.0,
    salience_action=True,
    visible_glimpse=False,

    dense_reward=True,
    reward_window=0.499,

    ablation='easy',
    log_name='grid_arithmetic',
    render_rollouts=None
)


config.update(alg_config)
config.update(env_config)


grid = dict(
    exploration_schedule=[0.1] + ["Poly(1.0, {}, end=0.1)".format(i) for i in 2**np.arange(10, 20)],
)


from dps.parallel.hyper import build_and_submit
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
clify.wrap_function(build_and_submit)(config, grid, n_param_settings=None, host_pool=host_pool)
