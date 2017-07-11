import numpy as np

import clify
from dps import cfg
from dps.utils import Config
from dps.parallel.submit_job import submit_job
from dps.parallel.hyper import build_search


config = Config(
    curriculum=[
        dict(T=10, shape=(2, 2), n_digits=3, upper_bound=True),
        dict(T=15, shape=(3, 3), n_digits=3, upper_bound=True),
        dict(T=25, shape=(4, 4), n_digits=3, upper_bound=True),
        dict(T=30, shape=(5, 5), n_digits=3, upper_bound=True),
    ],
    base=10,
    gamma=0.99,
    upper_bound=True,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    n_train=10000,
    n_val=500,

    max_cg_steps=10,
    max_line_search_steps=20,

    display_step=10,
    eval_step=10,
    max_steps=100000,
    patience=10000,
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

    noise_schedule=None,

    deadline=''
)


with config:
    cl_args = clify.wrap_object(cfg).parse()
    cfg.update(cl_args)

    distributions = dict(
        n_controller_units=[32, 64, 128],
        batch_size=[16, 32, 64, 128],
        entropy_schedule=['constant {}'.format(n) for n in 0.5**np.arange(10, step=2)] +
                         ['poly {} 100000 1e-6 1'.format(n) for n in 0.5**np.arange(10, step=2)],
        exploration_schedule=[
            'exp 1.0 100000 0.01',
            'exp 1.0 100000 0.1',
            'exp 10.0 100000 0.01',
            'exp 10.0 100000 0.1',
        ],
        test_time_explore=[1.0, 0.1],
        delta_schedule=[
            'constant 1e-2',
            'constant 1e-3',
            'constant 1e-4',
            'poly 1e-2 100000 1e-6 1',
            'poly 1e-3 100000 1e-6 1',
            'poly 1e-4 100000 1e-6 1',
        ],
    )

    alg = 'trpo'
    task = 'alt_arithmetic'

    if 1:
        n_param_settings = 50
        n_repeats = 5
        walltime = "14:30:00"
        cleanup_time = "00:30:00"
        time_slack = 30
    else:
        n_param_settings = 8
        n_repeats = 4
        walltime = "00:30:00"
        cleanup_time = "00:02:15"
        time_slack = 30
    ppn = 4

    job, archive_path = build_search(
        '/tmp/dps/search', 'trpo_search', n_param_settings, n_repeats,
        alg, task, True, distributions, config, use_time=1)

    submit_job(
        archive_path, 'map', '/tmp/dps/search/execution/',
        show_script=True, parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
        env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'), ppn=ppn, hosts=hosts,
        walltime=walltime, cleanup_time=cleanup_time, time_slack=time_slack, redirect=True)
