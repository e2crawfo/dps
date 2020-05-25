import numpy as np

from dps.utils import Config, load_system_config


DEFAULT_CONFIG = Config(
    env_name="default_env",
    exp_name="",

    seed=-1,

    curriculum=[{}],

    initial_step=0,
    initial_stage=0,

    # "stage,step" - used outside of any stages, to set initial_stage and initial_step for that stage.
    # provided for easy access from command line.
    start_from="0,0",

    load_path='-1',  # path or stage to load variables from.
    do_train=True,
    preserve_env=False,
    power_through=True,  # Whether to complete the entire curriculum, even if threshold not reached.
    robust=True,
    pdb=False,
    update_latest=True,
    variable_scope_depth=3,
    backup_dir=None,
    backup_step=5000,

    patience=np.inf,
    max_n_fallbacks=0,

    render_step=np.inf,
    display_step=100,
    eval_step=100,
    checkpoint_step=5000,
    weight_step=5000,
    overwrite_weights=True,
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
    max_stages=0,
    max_experiences=np.inf,

    render_hook=None,
    render_first=False,
    render_final=True,

    stopping_criteria="",

    tee=True,  # If True, output of training run (stdout and stderr) is written to screen as
               # well as a file. If False, only written to the file.

    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0,

    readme="",
    hooks=[],
    overwrite_plots=True,
    n_procs=1,
    profile=False,
    copy_dataset_to="",
    warning_mode='once',
    start_tensorboard=0,
)

DEFAULT_CONFIG.update(load_system_config())
