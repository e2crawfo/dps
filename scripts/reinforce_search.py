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

    display_step=1000,
    eval_step=10,
    max_steps=100000,
    patience=5000,
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
)


with config:
    cl_args = clify.wrap_object(cfg).parse()
    cfg.update(cl_args)

    distributions = dict(
        n_controller_units=[32, 64, 128],
        batch_size=[16, 32, 64, 128],
        entropy_schedule=['constant {}'.format(n) for n in 0.5**np.arange(1, 4, step=1)] +
                         ['poly {} 100000 1e-6 1'.format(n) for n in 0.5**np.arange(1, 4, step=1)],
        exploration_schedule=[
            'poly 1.0 100000 0.01 ',
            'poly 1.0 100000 0.1 1',
            'poly 10.0 100000 0.01 1',
            'poly 10.0 100000 0.1 1',
        ],
        test_time_explore=[1.0, 0.1, -1],
        lr_schedule=[
            'constant 1e-3',
            'constant 1e-4',
            'constant 1e-5',
            'poly 1e-3 100000 1e-6 1',
            'poly 1e-4 100000 1e-6 1',
            'poly 1e-5 100000 1e-6 1',
        ],
    )

    alg = 'reinforce'
    task = 'alt_arithmetic'
    # hosts = ['ecrawf6@lab1-{}.cs.mcgill.ca'.format(i+1) for i in range(10, 20)]
    # hosts = [":", "ecrawf6@lab1-1.cs.mcgill.ca"]
    hosts = [":"]

    if 0:
        # Big
        n_param_settings = 20
        n_repeats = 5
        walltime = "12:00:00"
        cleanup_time = "00:30:00"
        time_slack = 60
        ppn = 4
    elif 0:
        # Medium
        n_param_settings = 8
        n_repeats = 4
        walltime = "00:30:00"
        cleanup_time = "00:02:15"
        time_slack = 30
        ppn = 4
    else:
        # Small
        n_param_settings = 4
        n_repeats = 2
        walltime = "00:05:00"
        cleanup_time = "00:00:45"
        time_slack = 45
        ppn = 2

    job, archive_path = build_search(
        '/tmp/dps/search', 'reinforce_search', n_param_settings, n_repeats,
        alg, task, True, distributions, config, use_time=1)

    submit_job(
        "REINFORCE_SEARCH", archive_path, 'map', '/tmp/dps/search/execution/',
        parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
        env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'), ppn=ppn, hosts=hosts,
        walltime=walltime, cleanup_time=cleanup_time, time_slack=time_slack, redirect=True)
