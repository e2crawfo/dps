import numpy as np

import clify

from dps.utils import Config
from dps.utils.tf import LeNet
from dps.envs import grid_arithmetic
from dps.rl.algorithms import a2c
from dps.rl.policy import BuildEpsilonSoftmaxPolicy, BuildLstmController
from dps.rl import rl_render_hook
from dps.config import DEFAULT_CONFIG
from dps.vision import MNIST_CONFIG, MNIST_SALIENCE_CONFIG
from dps.test.test_mnist import salience_render_hook


config = DEFAULT_CONFIG.copy(
    name="GridArithmeticExperiments",

    n_train=10000,
    n_val=100,
    max_steps=1000000,
    display_step=10,
    eval_step=10,
    patience=np.inf,
    power_through=False,
    preserve_policy=True,

    slim=False,
    save_summaries=True,
    start_tensorboard=True,
    verbose=False,
    show_plots=True,
    save_plots=True,

    threshold=0.01,
    render_hook=rl_render_hook,
    memory_limit_mb=12*1024,
)


alg_config = Config(
    get_updater=a2c.A2C,
    build_policy=BuildEpsilonSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",

    policy_weight=1.0,
    value_reg_weight=0.0,
    value_weight=32.0,
    entropy_weight=0.5,

    lr_schedule=1e-4,
    n_controller_units=128,
    batch_size=16,
    gamma=0.98,
    opt_steps_per_update=10,
    epsilon=0.2,
    split=False,

    exploration_schedule="Poly(1.0, 0.1, 8192)",
    actor_exploration_schedule=None,
    val_exploration_schedule="0.0",

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

    # reductions="sum",
    # reductions="prod",
    # reductions="max",
    # reductions="min",
    reductions="A:sum M:prod X:max N:min",
    arithmetic_actions='+ * max min +1',

    curriculum=[
        dict(T=30, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    mnist=True,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.01,

    salience_shape=(2, 2),
    salience_action=True,
    visible_glimpse=False,
    initial_salience=False,
    salience_input_width=3*14,
    salience_output_width=14,

    reward_window=0.499,
    final_reward=True,
    downsample_factor=2,

    ablation='easy',
    log_name='grid_arithmetic',
    render_rollouts=None,

    build_digit_classifier=lambda: LeNet(128, scope="digit_classifier"),
    build_op_classifier=lambda: LeNet(128, scope="op_classifier"),

    mnist_config=MNIST_CONFIG.copy(
        eval_step=100,
        max_steps=100000,
        patience=5000,
        threshold=0.01,
        include_blank=True
    ),

    salience_config=MNIST_SALIENCE_CONFIG.copy(
        eval_step=100,
        max_steps=100000,
        patience=5000,
        threshold=0.001,
        render_hook=salience_render_hook(),
    ),
)

config.update(alg_config)
config.update(env_config)


# For oak experiment
config.update(
    reductions="sum",
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,
)


# grid = dict(entropy_weight=2**np.linspace(-4, 3, 8))
grid = dict(n_train=2**np.arange(6, 18))

from dps.parallel.hyper import build_and_submit
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=host_pool)
