import copy
import numpy as np
from pathlib import Path
import pandas as pd
from pprint import pprint
import time
import datetime
from collections import OrderedDict
from itertools import product
from copy import deepcopy
import os
from tabulate import tabulate
import shutil
import dill
import sys
import subprocess

import clify

from dps import cfg
from dps.utils import gen_seed, Config, cd
from dps.parallel.submit_job import ParallelSession
from dps.parallel.base import Job, ReadOnlyJob
from spectral_dagger.utils.experiment import ExperimentStore


def nested_sample(distributions, n_samples=1):
    assert isinstance(distributions, dict)
    config = Config(distributions)
    flat = config.flatten()
    dists = OrderedDict()
    other = {}

    for k, v in flat.items():
        try:
            v = list(v)
            dists[k] = v
        except (TypeError, ValueError):
            if hasattr(v, 'rvs'):
                dists[k] = v
            else:
                other[k] = v

    samples = []
    for k, v in dists.items():
        if hasattr(v, 'rvs'):
            samples.append(v.rvs(n_samples))
        else:
            samples.append(np.random.choice(v, size=n_samples))

    samples = zip(*samples)

    _samples = []
    for s in samples:
        new = Config(deepcopy(other.copy()))
        for k, item in zip(flat, s):
            new[k] = item
        _samples.append(type(distributions)(new))
    return _samples


def nested_map(d, f):
    if isinstance(d, dict):
        _d = d.copy()
        _d.update({k: nested_map(v, f) for k, v in d.items()})
        return _d
    else:
        return f(d)


def generate_all(distributions):
    assert isinstance(distributions, dict)
    config = Config(distributions)
    flat = config.flatten()
    lists = OrderedDict()
    other = {}

    for k, v in flat.items():
        try:
            v = list(v)
            lists[k] = v
        except (TypeError, ValueError):
            if hasattr(v, 'rvs'):
                raise Exception(
                    "Attempting to generate all samples, but element {} "
                    "with key {} is a continuous distribution.".format(v, k))
            other[k] = v

    cartesian_product = product(*lists.values())
    samples = []
    for p in cartesian_product:
        new = Config(deepcopy(other.copy()))
        for k, item in zip(flat, p):
            new[k] = item
        samples.append(type(distributions)(new))
    return samples


def sample_configs(distributions, base_config, n_repeats, n_samples=None):
    """
    Parameters
    ----------
    distributions: dict
        Mapping from parameter names to distributions (objects with
        member function ``rvs`` which accepts a shape and produces
        an array of samples with that shape).
    base_config: Config instance
        The base config, supplies any parameters not covered in ``distribution``.
    n_repeats: int > 0
        Number of different seeds to use for each sampled configuration.
    n_samples: int > 0
        Number of configs to sample.

    """
    samples = []

    if n_samples is None:
        samples = generate_all(distributions)
    else:
        samples = nested_sample(distributions, n_samples)

    configs = []
    for i, s in enumerate(samples):
        s['idx'] = i
        for r in range(n_repeats):
            _new = copy.deepcopy(s)
            _new['repeat'] = r
            _new['seed'] = gen_seed()
            configs.append(_new)

    return configs


class RunTrainingLoop(object):
    def __init__(self, base_config):
        self.base_config = base_config

    def __call__(self, new):
        os.nice(10)

        start_time = time.time()
        print("Starting new training run at: ")
        print(datetime.datetime.now())
        print("Sampled values: ")
        pprint(new)

        config = copy.copy(self.base_config)
        config.update(new)
        config.update(
            start_tensorboard=False,
            save_summaries=False,
            update_latest=False,
            display=False,
            save_display=False,
            max_experiments=np.inf,
        )

        with config:
            cl_args = clify.wrap_object(cfg).parse()
            config.update(cl_args)

            from dps.train import training_loop
            val = training_loop(start_time=start_time)

        return val


def build_search(
        path, name, distributions, config, n_repeats, n_param_settings=None,
        _zip=True, use_time=0, do_local_test=True):
    """ Create a Job implementing a hyper-parameter search.

    Parameters
    ----------
    path: str
        Path to the directory where the archive that is built for the search will be saved.
    name: str
        Name for the search.
    distributions: dict (str -> distribution)
        Distributions to sample from.
    config: Config instance
        The base configuration.
    n_repeats: int
        Number of different random seeds to run each sample with.
    n_param_settings: int
        Number of parameter settings to sample. If not supplied, all possibilities are generated.
    _zip: bool
        Whether to zip the created search directory.
    use_time: bool
        Whether to add time to name of experiment directory.
    do_local_test: bool
        If True, run a short test using one of the sampled
        aonfigs on the local machine to catch any dumb errors
        before starting the real experiment.

    """
    with config:
        cl_args = clify.wrap_object(cfg).parse()
        config.update(cl_args)

    es = ExperimentStore(str(path), max_experiments=10, delete_old=1)
    count = 0
    base_name = name
    has_built = False
    while not has_built:
        try:
            exp_dir = es.new_experiment(name, use_time=use_time, force_fresh=1)
            has_built = True
        except FileExistsError:
            name = "{}_{}".format(base_name, count)
            count += 1
    print(str(config))

    print("Building parameter search at {}.".format(exp_dir.path))

    job = Job(exp_dir.path)

    new_configs = sample_configs(distributions, config, n_repeats, n_param_settings)

    print("{} configs were sampled for parameter search.".format(len(new_configs)))

    new_configs = [Config(c).flatten() for c in new_configs]

    if do_local_test:
        print("\nStarting local test " + ("=" * 80))
        test_config = new_configs[0].copy()
        test_config['max_steps'] = 100
        test_config['visualize'] = False
        RunTrainingLoop(config)(test_config)
        print("Done local test " + ("=" * 80) + "\n")

    job.map(RunTrainingLoop(config), new_configs)

    job.save_object('metadata', 'distributions', distributions)
    job.save_object('metadata', 'config', config)

    if _zip:
        path = job.zip(delete=False)
    else:
        path = exp_dir.path

    return job, path


def _summarize_search(args):
    """ Get all completed jobs, get their outputs. Summarize em. """
    print("Summarizing search stored at {}.".format(Path(args.path).absolute()))
    job = ReadOnlyJob(args.path)
    distributions = job.objects.load_object('metadata', 'distributions')
    distributions = Config(distributions)
    keys = list(distributions.keys())

    records = []
    for op in job.completed_ops():
        if 'map' in op.name:
            try:
                r = op.get_outputs(job.objects)[0]
            except BaseException as e:
                print("Exception thrown when accessing output of op {}:\n    {}".format(op.name, e))

        record = r['history'][-1].copy()
        record['host'] = r['host']
        record['op_name'] = op.name
        del record['best_path']

        if len(record['train_data']) > 0:
            for k, v in record['train_data'].iloc[-1].items():
                record[k + '_train'] = v
        if len(record['update_data']) > 0:
            for k, v in record['update_data'].iloc[-1].items():
                record[k + '_update'] = v
        if len(record['val_data']) > 0:
            for k, v in record['val_data'].iloc[-1].items():
                record[k + '_val'] = v

        del record['train_data']
        del record['update_data']
        del record['val_data']

        config = Config(r['config'])
        for k in keys:
            record[k] = config[k]

        record.update(
            latest_stage=r['history'][-1]['stage'],
            total_steps=sum(s['n_steps'] for s in r['history']),
        )

        record['seed'] = r['config']['seed']
        records.append(record)

    df = pd.DataFrame.from_records(records)
    for key in keys:
        df[key] = df[key].fillna(-np.inf)

    groups = df.groupby(keys)

    data = []
    for k, _df in groups:
        _df = _df.sort_values(['latest_stage', 'best_loss'])
        data.append(dict(
            data=_df,
            keys=[k] if len(distributions) == 1 else k,
            latest_stage=_df.latest_stage.max(),
            stage_sum=_df.latest_stage.sum(),
            best_loss=_df.best_loss.mean()))

    data = sorted(data, reverse=False, key=lambda x: (x['latest_stage'], -x['best_loss'], x['stage_sum']))

    column_order = [
        'latest_stage', 'best_loss', 'seed', 'reason', 'total_steps', 'n_steps', 'host']
    remaining = [k for k in data[0]['data'].keys() if k not in column_order and k not in keys]
    column_order = column_order + sorted(remaining)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\n' + '*' * 100)
        print("RESULTS GROUPED BY PARAM VALUES, WORST COMES FIRST: ")
        for i, d in enumerate(data):
            print('\n {} '.format(len(data)-i) + '*' * 40)
            pprint({n: v for n, v in zip(keys, d['keys'])})
            _data = d['data'].drop(keys, axis=1)
            _data = _data[column_order]
            with_stats = pd.merge(
                _data.transpose(), _data.describe().transpose(),
                left_index=True, right_index=True, how='outer')
            print(tabulate(with_stats, headers='keys', tablefmt='fancy_grid'))

    print('\n' + '*' * 100)
    print("BASE CONFIG")
    print(job.objects.load_object('metadata', 'config'))

    print('\n' + '*' * 100)
    print("DISTRIBUTIONS")
    pprint(distributions)


def _zip_search(args):
    job = Job(args.to_zip)
    archive_name = args.name or Path(args.to_zip).stem
    job.zip(archive_name, delete=args.delete)


def hyper_search_cl():
    from dps.parallel.base import parallel_cl
    summary_cmd = (
        'summary', 'Summarize results of a hyper-parameter search.', _summarize_search,
        ('path', dict(help="Location of data store for job.", type=str)),
    )

    zip_cmd = (
        'zip', 'Zip up a job.', _zip_search,
        ('to_zip', dict(help="Path to the job we want to zip.", type=str)),
        ('name', dict(help="Optional path where archive should be created.", type=str, default='', nargs='?')),
        ('--delete', dict(help="If True, delete the original.", action='store_true'))
    )

    parallel_cl('Build, run and view hyper-parameter searches.', [summary_cmd, zip_cmd])


def build_and_submit(
        config, distributions, wall_time, cleanup_time, max_hosts=2, ppn=2,
        n_param_settings=2, n_repeats=2, host_pool=None, n_retries=1, do_local_test=False):

    build_params = dict(n_param_settings=n_param_settings, n_repeats=n_repeats)
    run_params = dict(
        wall_time=wall_time, cleanup_time=cleanup_time, time_slack=60,
        max_hosts=max_hosts, ppn=ppn, n_retries=n_retries, hpc=False,
        host_pool=host_pool)

    with config:
        job, archive_path = build_search(
            '/tmp/dps/search', config.name, distributions, config,
            use_time=1, _zip=True, do_local_test=do_local_test, **build_params)

        session = ParallelSession(
            config.name, archive_path, 'map', '/tmp/dps/search/execution/',
            parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
            env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'),
            redirect=True, **run_params)

        session.run()


def build_and_submit_hpc(
        config, distributions, wall_time, cleanup_time, max_hosts=2, ppn=2,
        n_param_settings=2, n_repeats=2, n_retries=1, do_local_test=False, queue=""):

    build_params = dict(n_param_settings=n_param_settings, n_repeats=n_repeats)
    run_params = dict(
        wall_time=wall_time, cleanup_time=cleanup_time, time_slack=60,
        max_hosts=max_hosts, ppn=ppn, n_retries=n_retries, hpc=True)

    with config:
        job, archive_path = build_search(
            '/tmp/dps/search', config.name, distributions, config,
            use_time=1, _zip=True, do_local_test=do_local_test, **build_params)

        session = ParallelSession(
            config.name, archive_path, 'map', '$SCRATCH', local_scratch_prefix="\\$RAMDISK",
            parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
            env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'),
            redirect=True, **run_params)

    os.remove(str(archive_path))
    shutil.rmtree(str(archive_path).split('.')[0])

    job_dir = Path(session.job_directory)

    python_script = """#!{}
import dill
with open("./session.pkl", "rb") as f:
    session = dill.load(f)
session.run()
""".format(sys.executable)
    with (job_dir / "run.py").open('w') as f:
        f.write(python_script)

    with (job_dir / "session.pkl").open('wb') as f:
        dill.dump(session, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    resources = "nodes={}:ppn={},walltime={}".format(session.n_nodes, session.ppn, session.wall_time_seconds)
    project = "jim-594-aa"
    email = "eric.crawford@mail.mcgill.ca"
    if queue:
        queue = "-q " + queue
    command = (
        "qsub -N {name} -d {job_dir} -w {job_dir} -m abe -M {email} -A {project} {queue} -V "
        "-l {resources} -e stderr.txt -o stdout.txt run.py".format(
            name=config.name, job_dir=job_dir, email=email, project=project,
            queue=queue, resources=resources
        )
    )

    with cd(job_dir):
        subprocess.run(command.split())
