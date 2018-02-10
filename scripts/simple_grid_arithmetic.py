import numpy as np

import clify

from dps.utils import Config
from dps.env import simple_grid_arithmetic
from dps.rl.algorithms import a2c
from dps.rl.policy import BuildEpsilonSoftmaxPolicy, BuildLstmController
from dps.config import RL_EXPERIMENT_CONFIG


env_config = simple_grid_arithmetic.config.copy(
    reductions="A:sum,M:prod,X:max,N:min",
    arithmetic_actions='+,*,max,min,+1',
    ablation='easy',
    render_rollouts=None,

    curriculum=[
        dict(),
    ],
    T=30,
    min_digits=2,
    max_digits=3,
    shape=(2, 2),
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.01,
    largest_digit=100,

    salience_action=True,
    salience_shape=(2, 2),
    visible_glimpse=False,
    initial_salience=False,

    final_reward=True,

    n_train=10000,
    n_val=100,
)

alg_config = Config(
    get_updater=a2c.A2C,
    build_policy=BuildEpsilonSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",

    policy_weight=1.0,
    entropy_weight=0.01,

    value_weight=1.0,
    value_reg_weight=0.0,
    value_epsilon=0,
    value_n_samples=0,
    value_direct=False,

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

config = RL_EXPERIMENT_CONFIG.copy(
    name="SimpleGridArithmeticRL",

    cpu_ram_limit_mb=12*1024,
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,
)

config.update(alg_config)
config.update(env_config)

grid = dict(n_train=2**np.arange(6, 18))

from dps.parallel.hyper import build_and_submit
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=host_pool)
