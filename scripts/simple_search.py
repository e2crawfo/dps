import numpy as np
from scipy.stats import distributions as spdists

import clify

from dps import cfg
from dps.parallel.submit_job import submit_job
from dps.parallel.hyper import build_search, ChoiceDist, LogUniform
from dps.config import algorithms, tasks

alg = 'reinforce'
task = 'room'

config = tasks[task]
config.update(algorithms[alg])

with config:
    cl_args = clify.wrap_object(cfg).parse()
    cfg.update(cl_args)

config.use_gpu = False
config.max_steps = 20
config.verbose = False
config.visualize = False
config.display = False
config.save_display = False

distributions = dict(
    lr_start=LogUniform(-3., 0., 1),
    exploration_start=spdists.uniform(0, 0.5),
    batch_size=ChoiceDist(10 * np.arange(1, 11)),
    scaled=ChoiceDist([0, 1]),
    entropy_start=ChoiceDist([0.0, LogUniform(-3., 0., 1)]),
    max_grad_norm=ChoiceDist([0.0, 1.0, 2.0])
)

job, archive_path = build_search(
    '/tmp/dps/search', 'simple_search', 3, 2, 'reinforce', 'room', True, distributions, config, use_time=1)

hosts = [
    ':',
    'ecrawf6@lab1-1.cs.mcgill.ca',
    'ecrawf6@lab1-2.cs.mcgill.ca',
]

submit_job(
    archive_path, 'map', '/tmp/dps/search/execution/', pbs=False,
    show_script=True, parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
    env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'), ppn=2, hosts=hosts)

submit_job(
    archive_path, 'reduce', '/tmp/dps/search/execution/', pbs=False,
    show_script=False, parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
    env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'), ppn=1)
