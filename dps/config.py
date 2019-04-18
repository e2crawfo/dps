import numpy as np

from dps import cfg
from dps.utils import SYSTEM_CONFIG
from dps.rl import rl_render_hook


DEFAULT_CONFIG = SYSTEM_CONFIG.copy(
    env_name="default_env",
    exp_name="",

    seed=-1,

    curriculum=[{}],

    load_path=-1,  # path or stage to load variables from.
    do_train=True,
    preserve_env=False,
    power_through=True,  # Whether to complete the entire curriculum, even if threshold not reached.
    robust=True,
    pdb=False,

    patience=np.inf,

    render_step=np.inf,
    display_step=100,
    eval_step=100,
    checkpoint_step=5000,
    store_step_data=True,

    n_train=10000,
    n_val=500,
    batch_size=16,
    reward_window=0.499,
    threshold=0.01,

    noise_schedule=None,
    max_grad_norm=None,

    max_time=0,
    max_steps=1000000,
    max_experiences=np.inf,

    render_hook=None,
    render_first=False,

    stopping_criteria="",

    tee=True,  # If True, output of training run (stdout and stderr) will is written to screen as
               # well as a file. If False, only written to the file.

    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0,

    readme="",
    hooks=[],
    overwrite_plots=True,
    n_procs=1,
    profile=False,
)


RL_EXPERIMENT_CONFIG = DEFAULT_CONFIG.copy(
    name="RLExperiment",

    render_n_rollouts=10,
    render_hook=rl_render_hook,

    standardize_advantage=True,
    n_controller_units=64,
    gamma=1.0,
    opt_steps_per_update=1,

    display_step=100,
    eval_step=100,
    patience=np.inf,
    max_steps=1000000,
    power_through=False,
)


SL_EXPERIMENT_CONFIG = RL_EXPERIMENT_CONFIG.copy(
    name="SLExperiment",
    patience=5000,
)


cfg._stack.append(DEFAULT_CONFIG)
