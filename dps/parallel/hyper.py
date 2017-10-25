import copy
import numpy as np
from pathlib import Path
import pandas as pd
from pprint import pprint
import time
import datetime
from collections import defaultdict
from itertools import product
from copy import deepcopy
import os
from tabulate import tabulate
import shutil
import dill
import sys
import subprocess
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO

import clify

from dps import cfg
from dps.utils.base import gen_seed, Config, cd, ExperimentStore
from dps.parallel.submit_job import ParallelSession
from dps.parallel.base import Job, ReadOnlyJob


def nested_map(d, f):
    if isinstance(d, dict):
        _d = d.copy()
        _d.update({k: nested_map(v, f) for k, v in d.items()})
        return _d
    else:
        return f(d)


def nested_sample(distributions, n_samples=1):
    assert isinstance(distributions, dict)
    config = Config(distributions)
    flat = config.flatten()
    other = {}
    samples = []

    sampled_keys = []

    for k in sorted(flat.keys()):
        v = flat[k]
        try:
            samples.append(list(np.random.choice(list(v), size=n_samples)))
        except (TypeError, ValueError):
            if hasattr(v, 'rvs'):
                samples.append(v.rvs(n_samples))
                sampled_keys.append(k)
            else:
                other[k] = v
        else:
            sampled_keys.append(k)

    samples = sorted(zip(*samples))

    configs = []
    for sample in samples:
        new = Config(deepcopy(other.copy()))
        for k, s in zip(sampled_keys, sample):
            new[k] = s
        configs.append(type(distributions)(new))
    return configs


def generate_all(distributions):
    assert isinstance(distributions, dict)
    config = Config(distributions)
    flat = config.flatten()
    other = {}

    sampled_keys = []
    lists = []

    for k in sorted(flat.keys()):
        v = flat[k]
        try:
            lists.append(list(v))
        except (TypeError, ValueError):
            if hasattr(v, 'rvs'):
                raise Exception(
                    "Attempting to generate all samples, but element {} "
                    "with key {} is a continuous distribution.".format(v, k))
            other[k] = v
        else:
            sampled_keys.append(k)

    param_sets = sorted(product(*lists))

    configs = []
    for pset in param_sets:
        new = Config(deepcopy(other.copy()))
        for k, p in zip(sampled_keys, pset):
            new[k] = p
        configs.append(type(distributions)(new))
    return configs


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

    if isinstance(distributions, list):
        samples = distributions + []
    else:
        if not n_samples:
            samples = generate_all(distributions)
        else:
            samples = nested_sample(distributions, n_samples)

    print("Sampled configs:")
    print(samples)

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
        Distributions to sample from. Can also be a list of samples.
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

    # Can't remember why I thought I needed this...
    # new_configs = [Config(c).flatten() for c in new_configs]

    if do_local_test:
        print("\nStarting local test " + ("=" * 80))
        test_config = new_configs[0].copy()
        test_config.update(
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


def process_detailed_data(record, kind):
    try:
        data = pd.read_csv(StringIO(record[kind + '_data']), index_col=False)
        if len(data) > 0:
            for k, v in data.iloc[-1].items():
                record[k + '_' + kind] = v
        del record[kind + '_data']
        del data
    except:
        pass


def _print_config(args):
    job = ReadOnlyJob(args.path)
    distributions = job.objects.load_object('metadata', 'distributions')
    distributions = Config(distributions)

    print("BASE CONFIG")
    print(job.objects.load_object('metadata', 'config'))

    print('\n' + '*' * 100)
    print("DISTRIBUTIONS")
    pprint(distributions)


def _summarize_search(args):
    """ Get all completed jobs, get their outputs. Summarize em. """
    print("Summarizing search stored at {}.".format(Path(args.path).absolute()))

    job = ReadOnlyJob(args.path)

    distributions = job.objects.load_object('metadata', 'distributions')
    if isinstance(distributions, list):
        keys = set()
        for d in distributions:
            keys |= set(d.keys())
        keys = list(keys)
    else:
        distributions = Config(distributions)
        keys = list(distributions.keys())
    keys = sorted(keys)

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

        process_detailed_data(record, 'train')
        process_detailed_data(record, 'update')
        process_detailed_data(record, 'val')
        process_detailed_data(record, 'test')

        config = Config(r['config'])
        for k in keys:
            try:
                record[k] = config[k]
            except KeyError:
                record[k] = None

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
    if isinstance(distributions, list):
        keys = set()
        for d in distributions:
            keys |= set(d.keys())
        keys = list(keys)
    else:
        distributions = Config(distributions)
        keys = list(distributions.keys())
    keys = sorted(keys)

    val_data = defaultdict(list)

    for op in job.completed_ops():
        if 'map' in op.name:
            try:
                r = op.get_outputs(job.objects)[0]
            except BaseException as e:
                print("Exception thrown when accessing output of op {}:\n    {}".format(op.name, e))

        config = Config(r['config'])
        key = ",".join("{}={}".format(k, config[k]) for k in keys)

        record = r['history'][-1].copy()

        vd = pd.read_csv(StringIO(record['val_data']), index_col=False)
        val_data[key].append(vd['loss'])

        del record['train_data']
        del record['update_data']
        del record['val_data']
        try:
            del record['test_data']
        except KeyError:
            pass
        del vd
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
        plt.savefig('rl_plot.pdf')


def ci(data, coverage):
    return stats.t.interval(
        coverage, len(data)-1, loc=np.mean(data), scale=stats.sem(data))


def _sample_complexity_plot(args):
    style = args.style
    spread_measure = args.spread_measure
    paths = args.paths
    paths = [Path(p).absolute() for p in paths]

    plt.title(args.title)

    label_order = []
    for p in paths:
        lo = _sample_complexity_plot_core(p, style, spread_measure)
        label_order.extend(lo)

    ax = plt.gca()
    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    if getattr(args, 'do_legend', False):
        ax.legend(
            ordered_handles, label_order, loc='center left',
            bbox_to_anchor=(1.05, 0.5), ncol=1)
    plt.grid(True)

    plt.ylim((0.0, 100.0))

    # plt.ylabel("% Incorrect on Test Set")
    # plt.xlabel("# Training Examples")
    # plt.subplots_adjust(
    #     left=0.09, bottom=0.13, right=0.7, top=0.93, wspace=0.05, hspace=0.18)
    # plt.show()
    if args.filename:
        plt.savefig('{}.pdf'.format(args.filename))


def _sample_complexity_plot_core(path, style, spread_measure):
    print("Plotting searches stored at {}.".format(path))

    job = ReadOnlyJob(path)
    distributions = job.objects.load_object('metadata', 'distributions')
    if isinstance(distributions, list):
        keys = set()
        for d in distributions:
            keys |= set(d.keys())
        keys = list(keys)
    else:
        distributions = Config(distributions)
        keys = list(distributions.keys())
    keys = sorted(keys)

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

        process_detailed_data(record, 'train')
        process_detailed_data(record, 'update')
        process_detailed_data(record, 'val')
        process_detailed_data(record, 'test')

        config = Config(r['config'])
        for k in keys:
            try:
                record[k] = config[k]
            except KeyError:
                record[k] = None

        record.update(
            latest_stage=r['history'][-1]['stage'],
            total_steps=sum(s['n_steps'] for s in r['history']),
        )

        record['seed'] = r['config']['seed']
        records.append(record)

    df = pd.DataFrame.from_records(records)
    for key in keys:
        df[key] = df[key].fillna(-np.inf)

    rl = 'n_controller_units' not in df

    if not rl:
        groups = df.groupby('n_controller_units')
        field_to_plot = 'test_reward'
    else:
        groups = [(128, df)]
        field_to_plot = 'test_loss'

    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    groups = sorted(groups, key=lambda x: x[0])

    label_order = []

    group_by_key = [k for k in keys if k.endswith('n_train')][0]

    with plt.style.context(style):
        for i, (k, _df) in enumerate(groups):
            _groups = _df.groupby(group_by_key)
            values = list(_groups)
            x = [v[0] for v in values]
            if rl:
                ys = [100 * v[1][field_to_plot] for v in values]
            else:
                ys = [-100 * v[1][field_to_plot] for v in values]

            y = [_y.mean() for _y in ys]

            if spread_measure == 'std_dev':
                y_upper = y_lower = [_y.std() for _y in ys]
            elif spread_measure == 'conf_int':
                conf_int = [ci(_y.values, 0.95) for _y in ys]
                y_lower = y - np.array([ci[0] for ci in conf_int])
                y_upper = np.array([ci[1] for ci in conf_int]) - y
            elif spread_measure == 'std_err':
                y_upper = y_lower = [stats.sem(_y.values) for _y in ys]
            else:
                raise Exception("NotImplemented")

            yerr = np.vstack((y_lower, y_upper))

            if rl:
                label = "RL - n_hidden_units={}".format(k)
                c = 'k'
            else:
                label = "CNN - n_hidden_units={}".format(k)
                c = colours[i % len(colours)]
            plt.semilogx(x, y, label=label, c=c, basex=2)
            label_order.append(label)
            plt.gca().errorbar(x, y, yerr=yerr, c=c)
    return label_order


def _zip_search(args):
    job = Job(args.to_zip)
    archive_name = args.name or Path(args.to_zip).stem
    job.zip(archive_name, delete=args.delete)


def dps_hyper_cl():
    from dps.parallel.base import parallel_cl
    config_cmd = (
        'config', 'Print config of a hyper-parameter search.', _print_config,
        ('path', dict(help="Location of data store for job.", type=str)),
    )

    summary_cmd = (
        'summary', 'Summarize results of a hyper-parameter search.', _summarize_search,
        ('path', dict(help="Location of data store for job.", type=str)),
    )

    style_list = ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')

    rl_plot_cmd = (
        'rl_plot', 'Plot results of an RL hyper-parameter search.', _rl_plot,
        ('paths', dict(help="Paths to locations of data stores.", type=str, default="results.zip", nargs='+')),
        ('--style', dict(help="Style for plot.", choices=style_list, default="ggplot")),
    )

    sc_plot_cmd = (
        'sc_plot', 'Plot sample complexity results.', _sample_complexity_plot,
        ('title', dict(help="Plot title.", type=str)),
        ('filename', dict(help="Plot filename.", type=str)),
        ('paths', dict(help="Paths to locations of data stores.", type=str, default="results.zip", nargs='+')),
        ('--style', dict(help="Style for plot.", choices=style_list, default="ggplot")),
        ('--spread-measure', dict(
            help="Measure of spread to use for error bars.", choices="std_dev conf_int std_err".split(),
            default="std_err")),
    )

    zip_cmd = (
        'zip', 'Zip up a job.', _zip_search,
        ('to_zip', dict(help="Path to the job we want to zip.", type=str)),
        ('name', dict(help="Optional path where archive should be created.", type=str, default='', nargs='?')),
        ('--delete', dict(help="If True, delete the original.", action='store_true'))
    )

    parallel_cl(
        'Build, run, plot and view results of hyper-parameter searches.',
        [config_cmd, summary_cmd, rl_plot_cmd, sc_plot_cmd, zip_cmd])


def build_and_submit(
        name, config, distributions=None, wall_time="1year", cleanup_time="1day", max_hosts=1, ppn=1,
        n_param_settings=0, n_repeats=1, n_retries=0, host_pool=None, pmem=0, queue="", do_local_test=False,
        kind="local", gpu_set="", step_time_limit="", ignore_gpu=False, do_cleanup=False):
    """ Meant to be called from within a script.

    Parameters
    ----------
    kind: str

    """
    assert kind in "pbs slurm parallel local".split()
    assert 'build_command' not in config
    config['build_command'] = ' '.join(sys.argv)
    print(config['build_command'])

    if kind == "local":
        with config:
            cl_args = clify.wrap_object(cfg).parse()
            config.update(cl_args)

            from dps.train import training_loop
            val = training_loop()
        return val
    else:
        config.name = name

        config = config.copy(
            start_tensorboard=False,
            save_summaries=False,
            update_latest=False,
            show_plots=False,
            slim=True,
            max_experiments=np.inf,
        )
        del config['log_root']
        del config['experiments_dir']
        del config['data_dir']
        del config['model_dir']

        with config:
            job, archive_path = build_search(
                cfg.experiments_dir, name, distributions, config,
                add_date=1, _zip=True, do_local_test=do_local_test,
                n_param_settings=n_param_settings, n_repeats=n_repeats)

        submit_job(**locals())

        os.remove(str(archive_path))
        shutil.rmtree(str(archive_path).split('.')[0])


def dps_submit_cl():
    clify.wrap_function(submit_job)()


def submit_job(
        archive_path, name, wall_time="1year", cleanup_time="1day", max_hosts=1, ppn=1,
        n_retries=0, host_pool=None, pmem=0, queue="", kind="local", gpu_set="",
        step_time_limit="", ignore_gpu=False, do_cleanup=False, **kwargs):

    os.nice(10)

    assert kind in "pbs slurm parallel".split()

    hpc = kind in "pbs slurm".split()

    if kind == "slurm" and not pmem:
        raise Exception("Must supply a value for pmem (per-process-memory in mb) when using SLURM")

    run_params = dict(
        wall_time=wall_time, cleanup_time=cleanup_time, time_slack=120 if hpc else 60,
        max_hosts=max_hosts, ppn=ppn, n_retries=n_retries, host_pool=host_pool, kind=kind,
        gpu_set=gpu_set, step_time_limit=step_time_limit, ignore_gpu=ignore_gpu, do_cleanup=do_cleanup)

    session = ParallelSession(
        name, archive_path, 'map', cfg.experiments_dir + '/execution/',
        parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
        env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'),
        redirect=True, **run_params)

    if kind == "parallel":
        session.run()
        return

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

    if kind == "pbs":
        resources = "nodes={}:ppn={},walltime={}".format(session.n_nodes, session.ppn, session.wall_time_seconds)
        if pmem:
            resources = "{},pmem={}mb".format(resources, pmem)

        project = "jim-594-aa"
        email = "eric.crawford@mail.mcgill.ca"
        if queue:
            queue = "-q " + queue
        command = (
            "qsub -N {name} -d {job_dir} -w {job_dir} -m abe -M {email} "
            "-A {project} {queue} -V -l {resources} "
            "-j oe output.txt run.py".format(
                name=name, job_dir=job_dir, email=email, project=project,
                queue=queue, resources=resources
            )
        )

    elif kind == "slurm":
        wall_time_minutes = int(np.ceil(session.wall_time_seconds / 60))
        resources = "--nodes={} --ntasks-per-node={} --time={}".format(
            session.n_nodes, session.ppn, wall_time_minutes)

        if pmem:
            resources = "{} --mem-per-cpu={}mb".format(resources, pmem)

        if gpu_set:
            n_gpus = len([int(i) for i in gpu_set.split(',')])
            resources = "{} --gres=gpu:{}".format(resources, n_gpus)

        project = "def-jpineau"
        email = "eric.crawford@mail.mcgill.ca"
        if queue:
            queue = "-p " + queue
        command = (
            "sbatch --job-name {name} -D {job_dir} --mail-type=ALL --mail-user=e2crawfo "
            "-A {project} {queue} --export=ALL {resources} "
            "-o output.txt run.py".format(
                name=name, job_dir=job_dir, email=email, project=project,
                queue=queue, resources=resources
            )
        )

    else:
        raise Exception()

    print(command)
    with cd(job_dir):
        subprocess.run(command.split())
