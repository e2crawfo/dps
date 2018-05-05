import numpy as np

from dps import cfg
from dps.utils import Config
from dps.rl import rl_render_hook


DEFAULT_CONFIG = Config(
    log_name="Default",
    seed=-1,

    curriculum=[{}],

    load_path="",  # Path to load variables from.
    do_train=True,
    preserve_policy=True,  # Whether to use the policy learned on the last stage of the curriculum for each new stage.
    load_stage=None,
    preserve_env=False,
    power_through=True,  # Whether to complete the entire curriculum, even if threshold not reached.
    robust=True,

    use_dataset_cache=False,

    patience=np.inf,

    render_step=np.inf,
    display_step=100,
    eval_step=100,
    checkpoint_step=5000,
    store_step_data=True,

    n_train=10000,
    n_val=500,
    batch_size=16,
    opt_steps_per_update=1,
    reward_window=0.499,
    threshold=0.01,

    gamma=1.0,
    noise_schedule=None,
    max_grad_norm=None,

    standardize_advantage=True,
    reset_env=True,

    n_controller_units=64,

    save_utils=False,

    max_time=0,
    max_steps=1000000,
    max_experiences=np.inf,

    render_n_rollouts=10,
    render_hook=rl_render_hook,

    stopping_criteria="",
    eval_mode="val",

    tee=True,  # If True, output of training run (stdout and stderr) will is written to screen as
               # well as a file. If False, only written to the file.

    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0,

    readme="",
    hooks=[],
)


RL_EXPERIMENT_CONFIG = DEFAULT_CONFIG.copy(
    name="RLExperiment",

    display_step=100,
    eval_step=100,
    patience=np.inf,
    max_steps=1000000,
    power_through=False,
    preserve_policy=True,
)


SL_EXPERIMENT_CONFIG = RL_EXPERIMENT_CONFIG.copy(
    name="SLExperiment",
    patience=5000,
)


cfg._stack.append(DEFAULT_CONFIG)
