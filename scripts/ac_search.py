import argparse
import numpy as np

import clify

from dps import cfg
from dps.parallel.submit_job import submit_job
from dps.parallel.hyper import build_search


def search(config, distributions, hosts=None):
    with config:
        cl_args = clify.wrap_object(cfg).parse()
        cfg.update(cl_args)

        parser = argparse.ArgumentParser()
        parser.add_argument('size')
        parser.add_argument('walltime')
        args = parser.parse_args()
        walltime = args.walltime
        size = args.size

        if hosts is not None:
            time_slack = 60
        else:
            time_slack = 30

        if size == 'big':
            # Big
            n_param_settings = 50
            n_repeats = 5
            cleanup_time = "00:30:00"
        elif size == 'medium':
            # Medium
            n_param_settings = 8
            n_repeats = 4
            cleanup_time = "00:02:15"
        elif size == 'small':
            # Small
            n_param_settings = 2
            n_repeats = 2
            cleanup_time = "00:00:45"
        else:
            raise Exception("Unknown size: `{}`.".format(size))
        ppn = 2

        job, archive_path = build_search(
            '/tmp/dps/search', config.name, n_param_settings, n_repeats,
            distributions, True, config, use_time=1)

        submit_job(
            config.name, archive_path, 'map', '/tmp/dps/search/execution/',
            parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
            env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'), ppn=ppn, hosts=hosts,
            walltime=walltime, cleanup_time=cleanup_time, time_slack=time_slack, redirect=True)
