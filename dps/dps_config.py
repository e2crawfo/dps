import numpy as np

config = dict(
    env_name="default_env",
    exp_name="",
    scratch_dir="./dps_scratch",
    make_dirs=True,

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

    render_step=5000,
    display_step=5000,
    eval_step=5000,
    checkpoint_step=5000,
    weight_step=5000,
    overwrite_weights=True,
    store_step_data=True,
    copy_dataset_to="",

    n_train=10000,
    n_val=500,
    batch_size=16,

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
    threshold=0.01,

    tee=True,  # If True, output of training run (stdout and stderr) is written to screen as
               # well as a file. If False, only written to the file.

    use_gpu=False,

    readme="",
    hooks=[],
    overwrite_plots=True,
    n_procs=1,
    profile=False,
    warning_mode='once',

    start_tensorboard=True,
    tbport=6006,
    reload_interval=10,

    show_plots=False,

    in_parallel_session=False,

    verbose=False,
    ssh_hosts=[],
    ssh_options=(
        "-oPasswordAuthentication=no "
        "-oStrictHostKeyChecking=no "
        "-oConnectTimeout=5 "
        "-oServerAliveInterval=2"
    ),

    # tf only
    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0,
    per_process_gpu_memory_fraction=0,
    gpu_allow_growth=True,
)
