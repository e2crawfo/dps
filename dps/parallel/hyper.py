import copy
import numpy as np
from pathlib import Path
import pandas as pd
from pprint import pprint
import time
import datetime
from collections import OrderedDict, defaultdict
from itertools import product
from copy import deepcopy
import os
from tabulate import tabulate
import shutil
import dill
import sys
import subprocess
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import sem

import clify

from dps import cfg
from dps.utils.base import gen_seed, Config, cd, ExperimentStore
from dps.parallel.submit_job import ParallelSession
from dps.parallel.base import Job, ReadOnlyJob


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

        with config:
            cl_args = clify.wrap_object(cfg).parse()
            config.update(cl_args)

            from dps.train import training_loop
            val = training_loop(start_time=start_time)

        return val


def build_search(
        path, name, distributions, config, n_repeats, n_param_settings=None,
        _zip=True, add_date=0, do_local_test=True):
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
    add_date: bool
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
            exp_dir = es.new_experiment(name, add_date=add_date, force_fresh=1)
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
        test_config = new_configs[0].copy(
            max_steps=1000,
            render_hook=None
        )
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

    with pd.option_context('show_plots.max_rows', None, 'show_plots.max_columns', None):
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


def _rl_plot(args):
    style = args.style
    paths = args.paths
    paths = [Path(p).absolute() for p in paths]
    print("Plotting searches stored at {}.".format(paths))

    if len(paths) > 1:
        raise Exception("Not implemented")

    path = paths[0]

    job = ReadOnlyJob(path)
    distributions = job.objects.load_object('metadata', 'distributions')
    distributions = Config(distributions)
    keys = sorted(list(distributions.keys()))

    val_data = defaultdict(list)

    for op in job.completed_ops():
        if 'map' in op.name:
            try:
                r = op.get_outputs(job.objects)[0]
            except BaseException as e:
                print("Exception thrown when accessing output of op {}:\n    {}".format(op.name, e))

        config = Config(r['config'])
        key = ",".join("{}={}".format(k, config[k]) for k in keys)

        val_data[key].append(r['history'][-1]['val_data']['RLContext:loss'])
        del op
        del r

    n_plots = len(val_data) + 1
    w = int(np.ceil(np.sqrt(n_plots)))
    h = int(np.ceil(n_plots / w))

    with plt.style.context(style):
        fig, axes = plt.subplots(h, w, sharex=True, sharey=True, figsize=(15, 10))
        final_ax = axes[-1, -1]

        for n, key in enumerate(sorted(val_data)):
            i = int(n / w)
            j = n % w
            ax = axes[i, j]
            for vd in val_data[key]:
                ax.plot(vd)
            ax.set_title(key)
            mean = pd.concat(val_data[key], axis=1).mean(axis=1)
            final_ax.plot(mean, label=key)

        legend_handles = {l: h for h, l in zip(*final_ax.get_legend_handles_labels())}
        ordered_labels = sorted(legend_handles.keys())
        ordered_handles = [legend_handles[l] for l in ordered_labels]

        final_ax.legend(
            ordered_handles, ordered_labels, loc='center left',
            bbox_to_anchor=(1.05, 0.5), ncol=1)

        plt.subplots_adjust(
            left=0.05, bottom=0.05, right=0.86, top=0.97, wspace=0.05, hspace=0.18)

        plt.show()


def _sample_complexity_plot(args):
    style = args.style
    spread_measure = args.spread_measure

    paths = args.paths
    paths = [Path(p).absolute() for p in paths]
    print("Plotting searches stored at {}.".format(paths))

    if len(paths) > 1:
        raise Exception("Not implemented")

    path = paths[0]

    job = ReadOnlyJob(path)
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

    groups = df.groupby('n_controller_units')

    field_to_plot = 'test_reward'
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    groups = sorted(groups, key=lambda x: x[0])

    label_order = []

    with plt.style.context(style):
        for i, (k, _df) in enumerate(groups):
            _groups = _df.groupby('n_train')
            values = list(_groups)
            x = [v[0] for v in values]
            ys = [-100 * v[1][field_to_plot] for v in values]

            y = [_y.mean() for _y in ys]

            if spread_measure == 'std_dev':
                y_upper = y_lower = [_y.std() for _y in ys]
            elif spread_measure == 'conf_int':
                conf_int = [sms.DescrStatsW(_y.values).tconfint_mean() for _y in ys]
                y_lower = y - np.array([ci[0] for ci in conf_int])
                y_upper = np.array([ci[1] for ci in conf_int]) - y
            elif spread_measure == 'std_err':
                y_upper = y_lower = [sem(_y.values) for _y in ys]
            else:
                pass

            yerr = np.vstack((y_lower, y_upper))

            c = colours[i % len(colours)]
            label = "n_hidden_units={}".format(k)
            plt.semilogx(x, y, label=label, c=c, basex=2)
            label_order.append(label)
            plt.gca().errorbar(x, y, yerr=yerr, c=c)

        ax = plt.gca()
        legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
        ordered_handles = [legend_handles[l] for l in label_order]
        ax.legend(ordered_handles, label_order, loc='upper right', ncol=1)
        plt.grid(True)

        plt.ylim((0.0, 100.0))

        plt.ylabel("% Incorrect on Test Set")
        plt.xlabel("# Training Examples")
        plt.show()
        plt.savefig('cnn_results.png')


def _zip_search(args):
    job = Job(args.to_zip)
    archive_name = args.name or Path(args.to_zip).stem
    job.zip(archive_name, delete=args.delete)


def dps_hyper_cl():
    from dps.parallel.base import parallel_cl
    summary_cmd = (
        'summary', 'Summarize results of a hyper-parameter search.', _summarize_search,
        ('paths', dict(help="Location of data store for job.", type=str)),
    )

    style_list = ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')

    rl_plot_cmd = (
        'rl_plot', 'Plot results of an RL hyper-parameter search.', _rl_plot,
        ('paths', dict(help="Paths to locations of data stores.", type=str, default="results.zip", nargs='+')),
        ('--style', dict(help="Style for plot.", choices=style_list, default="ggplot")),
    )

    sc_plot_cmd = (
        'sc_plot', 'Plot sample complexity results.', _sample_complexity_plot,
        ('paths', dict(help="Paths to locations of data stores.", type=str, default="results.zip", nargs='+')),
        ('--style', dict(help="Style for plot.", choices=style_list, default="ggplot")),
        ('--spread-measure', dict(
            help="Measure of spread to use for error bars.", choices="std_dev conf_int std_err".split(),
            default="std_dev")),
    )

    zip_cmd = (
        'zip', 'Zip up a job.', _zip_search,
        ('to_zip', dict(help="Path to the job we want to zip.", type=str)),
        ('name', dict(help="Optional path where archive should be created.", type=str, default='', nargs='?')),
        ('--delete', dict(help="If True, delete the original.", action='store_true'))
    )

    parallel_cl(
        'Build, run, plot and view results of hyper-parameter searches.',
        [summary_cmd, rl_plot_cmd, sc_plot_cmd, zip_cmd])


def build_and_submit(
        name, config, distributions=None, wall_time="1year", cleanup_time="1day", max_hosts=2, ppn=2,
        n_param_settings=2, n_repeats=2, host_pool=None, n_retries=1, pmem=0, queue="", do_local_test=False,
        hpc=False, local_only=False):

    os.nice(10)

    if local_only:
        with config:
            cl_args = clify.wrap_object(cfg).parse()
            config.update(cl_args)

            from dps.train import training_loop
            val = training_loop()
        return val
    else:
        config = config.copy(
            start_tensorboard=False,
            save_summaries=False,
            update_latest=False,
            show_plots=False,
            max_experiments=np.inf,
        )

        kwargs = locals()

        if hpc:
            return _build_and_submit_hpc(**locals())
        else:
            return _build_and_submit(**locals())


def _build_and_submit(
        name, config, distributions=None, wall_time="1year", cleanup_time="1day", max_hosts=2, ppn=2,
        n_param_settings=2, n_repeats=2, host_pool=None, n_retries=1, do_local_test=False, **kwargs):

    build_params = dict(n_param_settings=n_param_settings, n_repeats=n_repeats)
    run_params = dict(
        wall_time=wall_time, cleanup_time=cleanup_time, time_slack=60,
        max_hosts=max_hosts, ppn=ppn, n_retries=n_retries, hpc=False,
        host_pool=host_pool)

    config.name = name

    with config:
        job, archive_path = build_search(
            '/tmp/dps/search', name, distributions, config,
            add_date=1, _zip=True, do_local_test=do_local_test, **build_params)

        session = ParallelSession(
            name, archive_path, 'map', '/tmp/dps/search/execution/',
            parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
            env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'),
            redirect=True, **run_params)

        session.run()


def _build_and_submit_hpc(
        name, config, distributions, wall_time, cleanup_time, max_hosts=2, ppn=2,
        n_param_settings=2, n_repeats=2, n_retries=0, do_local_test=False, pmem=0, queue="", **kwargs):

    build_params = dict(n_param_settings=n_param_settings, n_repeats=n_repeats)
    run_params = dict(
        wall_time=wall_time, cleanup_time=cleanup_time, time_slack=120,
        max_hosts=max_hosts, ppn=ppn, n_retries=n_retries, hpc=True)

    config.name = name
    if pmem:
        config.memory_limit_mb = int(pmem)

    with config:
        job, archive_path = build_search(
            '/tmp/dps/search', name, distributions, config,
            add_date=1, _zip=True, do_local_test=do_local_test, **build_params)

        session = ParallelSession(
            name, archive_path, 'map', '$SCRATCH/experiments', local_scratch_prefix="\\$RAMDISK",
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
    if pmem:
        resources = "{},pmem={}mb".format(resources, pmem)

    project = "jim-594-aa"
    email = "eric.crawford@mail.mcgill.ca"
    if queue:
        queue = "-q " + queue
    command = (
        "qsub -N {name} -d {job_dir} -w {job_dir} -m abe -M {email} -A {project} {queue} -V "
        "-l {resources} -e stderr.txt -o stdout.txt run.py".format(
            name=name, job_dir=job_dir, email=email, project=project,
            queue=queue, resources=resources
        )
    )

    with cd(job_dir):
        subprocess.run(command.split())
