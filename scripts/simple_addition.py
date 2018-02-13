import numpy as np

import clify

from dps.utils import Config
from dps.env import simple_addition
from dps.rl.algorithms import a2c
from dps.rl.policy import BuildEpsilonSoftmaxPolicy, BuildLstmController
from dps.config import DEFAULT_CONFIG


config = DEFAULT_CONFIG.copy(
    name="SimpleAddition",

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

    use_gpu=False,
    threshold=0.01,
    # render_hook=rl_render_hook,
    render_hook=None,
    cpu_ram_limit_mb=5*1024,
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
    opt_steps_per_update=1,
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


env_config = simple_addition.config.copy(
    curriculum=[dict(width=1)],
)

config.update(alg_config)
config.update(env_config)

config.update(
    use_gpu=True,
    gpu_allow_growth=True,
)

grid = dict(n_train=2**np.arange(6, 18))

from dps.hyper import build_and_submit
host_pool = [':'] + ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=host_pool)
