import copy
import numpy as np
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
import inspect

import clify

import dps
from dps import cfg
from dps.utils.base import (
    gen_seed, Config, cd, ExperimentStore, edit_text, process_path)
from dps.parallel.submit_job import ParallelSession, DEFAULT_HOST_POOL
from dps.parallel.base import Job, ReadOnlyJob

default_host_pool = DEFAULT_HOST_POOL


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


def sample_configs(distributions, n_repeats, n_samples=None):
    """
    Parameters
    ----------
    distributions: dict
        Mapping from parameter names to distributions (objects with
        member function ``rvs`` which accepts a shape and produces
        an array of samples with that shape).
    n_repeats: int > 0
        Number of different seeds to use for each sampled configuration.
    n_samples: int > 0
        Number of configs to sample.

    """
    samples = []

    if isinstance(distributions, list):
        samples = distributions + []

        if n_samples:
            samples = list(np.random.permutation(samples)[:n_samples])
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


def sanitize_string(s):
    return str(s).replace('_', '-').replace(' ', '').replace('/', ',')


class RunTrainingLoop(object):
    """ Entry point for each process. """

    def __init__(self, base_config):
        self.base_config = base_config

    def __call__(self, new):
        os.nice(10)

        start_time = time.time()

        print("Starting new training run at: ")
        print(datetime.datetime.now())

        print("Sampled values: ")
        print(new)

        print("Base config: ")
        print(self.base_config)

        keys = [k for k in sorted(new.keys()) if k not in 'seed idx repeat'.split()]
        exp_name = '_'.join(
            "{}={}".format(sanitize_string(k), sanitize_string(new[k]))
            for k in keys)

        dps.reset_config()

        config = copy.copy(self.base_config)
        config.update(new)

        with config:
            cfg.update_from_command_line()

            from dps.train import stepped_training_loop
            yield from stepped_training_loop(exp_name=exp_name, start_time=start_time)


def build_search(
        path, name, distributions, config, n_repeats, n_param_settings=None,
        _zip=True, add_date=0, do_local_test=True, readme=""):
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
        configs on the local machine to catch any dumb errors
        before starting the real experiment.
    readme: str
        String specifiying context/purpose of search.

    """
    es = ExperimentStore(str(path), max_experiments=None, delete_old=1)

    count = 0
    base_name = name
    has_built = False
    while not has_built:
        try:
            exp_dir = es.new_experiment(name, config.seed, add_date=add_date, force_fresh=1)
            has_built = True
        except FileExistsError:
            name = "{}_{}".format(base_name, count)
            count += 1

    if readme:
        with open(exp_dir.path_for('README.md'), 'w') as f:
            f.write(readme)

    print(str(config))
    exp_dir.record_environment(config=config, git_modules=[dps])

    print("Building parameter search at {}.".format(exp_dir.path))

    job = Job(exp_dir.path)

    new_configs = sample_configs(distributions, n_repeats, n_param_settings)

    print("{} configs were sampled for parameter search.".format(len(new_configs)))

    if do_local_test:
        print("\nStarting local test " + ("=" * 80))
        test_config = new_configs[0].copy()
        test_config.update(max_steps=1000, render_hook=None)
        RunTrainingLoop(config)(test_config)
        print("Done local test " + ("=" * 80) + "\n")

    job.map(RunTrainingLoop(config.copy()), new_configs)

    job.save_object('metadata', 'distributions', distributions)
    job.save_object('metadata', 'config', config)

    if _zip:
        path = job.zip(delete=True)
    else:
        path = exp_dir.path
    return job, path


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

    print("Summarizing search stored at {}.".format(os.path.realpath(args.path)))

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

    df = extract_data_from_job(job, keys, as_frame=True, omit_timestep_data=True)

    groups = df.groupby(keys)

    data = []
    for k, _df in groups:
        _df = _df.sort_values(['latest_stage', 'best_stopping_criteria'])
        data.append(dict(
            data=_df,
            keys=[k] if len(distributions) == 1 else k,
            latest_stage=_df.latest_stage.max(),
            stage_sum=_df.latest_stage.sum(),
            best_stopping_criteria=_df.best_stopping_criteria.mean()))

    data = sorted(data, reverse=False, key=lambda x: (x['latest_stage'], x['best_stopping_criteria'], x['stage_sum']))

    column_order = [c for c in [
        'latest_stage', 'best_stopping_criteria', 'seed', 'reason', 'total_steps', 'n_steps', 'host'] if c in data]
    remaining = [k for k in data[0]['data'].keys() if k not in column_order and k not in keys]
    column_order = column_order + sorted(remaining)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\n' + '*' * 100)
        print("RESULTS GROUPED BY PARAM VALUES, ORDER OF INCREASING VALUE OF <best_stopping_criteria>: ")
        for i, d in enumerate(data):
            print('\n {} '.format(len(data)-i) + '*' * 40)
            pprint({n: v for n, v in zip(keys, d['keys'])})
            _data = d['data'].drop(keys, axis=1)
            _data = _data[column_order]
            with_stats = pd.merge(
                _data.transpose(), _data.describe().transpose(),
                left_index=True, right_index=True, how='outer')

            profile_rows = [k for k in with_stats.index if 'time' in k or 'duration' in k or 'memory' in k]
            other_rows = [k for k in with_stats.index if k not in profile_rows]

            print(tabulate(with_stats.loc[profile_rows], headers='keys', tablefmt='fancy_grid'))
            print(tabulate(with_stats.loc[other_rows], headers='keys', tablefmt='fancy_grid'))

    if not args.no_config:
        print('\n' + '*' * 100)
        print("BASE CONFIG")
        print(job.objects.load_object('metadata', 'config'))

        print('\n' + '*' * 100)
        print("DISTRIBUTIONS")
        pprint(distributions)

    print(job.summary(verbose=args.verbose))


def _traj_plot(args):
    style = args.style
    paths = args.paths
    paths = [process_path(p) for p in paths]

    kind = args.kind
    assert kind in "train val test update".split()
    field = args.field

    print(
        "Plotting {} value of field {} from experiments "
        "stored at {}.".format(kind, field, paths))

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

    data = defaultdict(list)

    first = True

    for op in job.completed_ops(partial=True):
        if 'map' in op.name:
            try:
                r = op.get_outputs(job.objects)[0]
            except BaseException as e:
                print("Exception thrown when accessing output of op {}:\n    {}".format(op.name, e))

        config = Config(r['config'])

        record = r['history'][-1].copy()

        trajectory = pd.read_csv(
            StringIO(record[kind + '_data']),
            index_col=False)

        if first:
            valid_fields = list(trajectory.keys())
            print("Valid fields are {}".format(valid_fields))
            first = False

        key = tuple(config[k] for k in keys)
        data[key].append(trajectory[field])

        del record['train_data']
        del record['update_data']
        del record['val_data']
        try:
            del record['test_data']
        except KeyError:
            pass
        del trajectory
        del op
        del r

    n_plots = len(data) + 1
    w = int(np.ceil(np.sqrt(n_plots)))
    h = int(np.ceil(n_plots / w))

    with plt.style.context(style):
        fig, axes = plt.subplots(h, w, sharex=True, sharey=True, figsize=(15, 10))
        axes = np.atleast_2d(axes)
        final_ax = axes[-1, -1]

        label_order = []

        for n, key in enumerate(sorted(data)):
            label = ",".join("{}={}".format(k, v) for k, v in zip(keys, key))
            label_order.append(label)

            i = int(n / w)
            j = n % w
            ax = axes[i, j]
            for vd in data[key]:
                ax.plot(vd)
            ax.set_title(label)
            mean = pd.concat(data[key], axis=1).mean(axis=1)
            final_ax.plot(mean, label=label)

        legend_handles = {l: h for h, l in zip(*final_ax.get_legend_handles_labels())}
        ordered_handles = [legend_handles[l] for l in label_order]

        final_ax.legend(
            ordered_handles, label_order, loc='center left',
            bbox_to_anchor=(1.05, 0.5), ncol=1)

        plt.subplots_adjust(
            left=0.05, bottom=0.05, right=0.86, top=0.97, wspace=0.05, hspace=0.18)

        plt.savefig('traj_plot_kind={}_field={}'.format(kind, field))
        plt.show()


def ci(data, coverage):
    return stats.t.interval(
        coverage, len(data)-1, loc=np.mean(data), scale=stats.sem(data))


def extract_data_from_job(job, data_keys=None, as_frame=True, omit_timestep_data=True):
    """ Extract a pandas data frame or list of dictionaries from a job.

    Parameters
    ----------
    job: string or Job instance
        If string, then a path to a directory reprenting the job to extract data from.
        Otherwise, a Job instance representing the path to extract data from.
    data_keys: list of string
        Keys to extract from the sub-job configurations and inject into the resulting data frames.
    as_frame: bool
        If True, returns that data as a dataframe.
        Otherwise, returns the data as a list of dictionaries
    omit_timestep_data: bool
        If True, per-timestep data is deleted to save RAM. Can only be set to False
        if `as_frame` is False.

    """
    if isinstance(job, str):
        job = ReadOnlyJob(job)

    if as_frame and not omit_timestep_data:
        raise Exception("If `omit_timestep_data` is False, `as_frame` must also be False.")

    kinds = 'train update val test'.split()

    if not data_keys:
        data_keys = []
    elif isinstance(data_keys, str):
        data_keys = data_keys.split()

    records = []
    for op in job.completed_ops(partial=True):
        if 'map' in op.name:
            try:
                r = op.get_outputs(job.objects)[0]
                print(r)
            except BaseException as e:
                print("Exception thrown when accessing "
                      "output of op {}:\n    {}".format(op.name, e))
                raise

        record = r['history'][-1].copy()
        record['host'] = r['host']
        record['op_name'] = op.name
        try:
            del record['best_path']
        except Exception:
            pass

        if omit_timestep_data:
            for k in kinds:
                try:
                    del record[k + '_data']
                except Exception:
                    pass
        else:
            for k in kinds:
                key = k + '_data'
                data = record.get(key, '').strip()
                if data:
                    record[key] = pd.read_csv(StringIO(data), index_col=False)

        config = Config(r['config'])
        for k in data_keys:
            try:
                record[k] = config[k]
            except KeyError:
                record[k] = None

        record.update(
            latest_stage=r['history'][-1]['stage'],
        )
        try:
            record.update(
                total_steps=sum(s['n_steps'] for s in r['history'])
            )
        except Exception:
            pass

        record['seed'] = r['config']['seed']
        records.append(record)

    if as_frame:
        df = pd.DataFrame.from_records(records)
        for key in data_keys:
            df[key] = df[key].fillna(-np.inf)
        return df
    else:
        return records


def _sample_complexity_plot(args):
    style = args.style
    spread_measure = args.spread_measure
    paths = args.paths
    paths = [process_path(p) for p in paths]

    plt.title(args.title)

    label_order = []
    for p in paths:
        lo = _sample_complexity_plot_core(p, style, spread_measure)
        label_order.extend(lo)

    ax = plt.gca()

    if getattr(args, 'do_legend', False):
        legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
        ordered_handles = [legend_handles[l] for l in label_order]
        ax.legend(
            ordered_handles, label_order, loc='center left',
            bbox_to_anchor=(1.05, 0.5), ncol=1)
    plt.ylim((0.0, 100.0))

    if args.filename:
        plt.savefig('{}'.format(args.filename))

    # plt.ylabel("% Incorrect on Test Set")
    # plt.xlabel("# Training Examples")
    # plt.subplots_adjust(
    #     left=0.09, bottom=0.13, right=0.7, top=0.93, wspace=0.05, hspace=0.18)
    # plt.show()


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

    df = extract_data_from_job(job, keys)

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


def _ssh_execute(command, host):
    ssh_options = (
        "-oPasswordAuthentication=no "
        "-oStrictHostKeyChecking=no "
        "-oConnectTimeout=5 "
        "-oServerAliveInterval=2"
    )
    cmd = "ssh {ssh_options} -T {host} \"{command}\"".format(ssh_options=ssh_options, host=host, command=command)
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _probe_hosts(args):
    n_connected = 0
    n_idle = 0

    for host in DEFAULT_HOST_POOL:
        if host is not ':':
            print("\n" + "*" * 80)
            print("Testing connection to host {}...".format(host))
            p = _ssh_execute("echo Connected to \$HOSTNAME", host)
            if p.returncode:
                print("Could not connect:")
                print(p.stdout.decode())
            else:
                n_connected += 1
                print("\nTOP:")
                p = _ssh_execute("top -bn2 | head -n 5", host)
                top_output = p.stdout.decode()
                print(top_output)

                cpu = top_output.split('\n')[2]
                start = cpu.find('ni')
                end = cpu.find('id')
                idle_cpu = float(cpu[start:end].split()[1])

                if idle_cpu > 95:
                    n_idle += 1

                print("\nWHO:")
                p = _ssh_execute("who", host)
                print(p.stdout.decode())

    print("Was able to connect to {} hosts.".format(n_connected))
    print("{} of those hosts have idle cpu percent > 95.".format(n_idle))


def dps_hyper_cl():
    from dps.parallel.base import parallel_cl
    config_cmd = (
        'config', 'Print config of a hyper-parameter search.', _print_config,
        ('path', dict(help="Location of data store for job.", type=str)),
    )

    summary_cmd = (
        'summary', 'Summarize results of a hyper-parameter search.', _summarize_search,
        ('path', dict(help="Location of data store for job.", type=str)),
        ('--no-config', dict(help="If supplied, don't print out config.", action='store_true')),
    )

    style_list = ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')

    traj_plot_cmd = (
        'traj_plot', 'Plot the trajectory of a value throughout all training runs.', _traj_plot,
        ('paths', dict(help="Paths to locations of data stores.", type=str, default="results.zip", nargs='+')),
        ('kind', dict(help="Mode to gather data from.", choices="train val test update".split(), default="")),
        ('field', dict(help="Field to plot.", default="")),
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

    probe_hosts_cmd = (
        'probe_hosts', 'Check the status of the hosts in host_pool.', _probe_hosts,
    )

    parallel_cl(
        'Build, run, plot and view results of hyper-parameter searches.',
        [config_cmd, summary_cmd, traj_plot_cmd, sc_plot_cmd, probe_hosts_cmd])


def build_and_submit(
        name, config, distributions=None, wall_time="1year", n_param_settings=0, n_repeats=1,
        do_local_test=False, kind="local", readme="", **run_kwargs):
    """ Meant to be called from within a script.

    Parameters
    ----------
    name: str
        Name of the experiment.
    config: Config instance or dict
        Configuration to use as the base config for all jobs.
    distributions: dict
        Object used to generate variations of the base config (so that different jobs test different parameters).
    wall_time: str
        Maximum amount of time that is to be used to complete jobs.
    n_param_settings: int
        Number of different configurations to sample from `distributions`. If not supplied, it is assumed
        that `distributions` actually specifies a grid search, and an attempt is made to generate
        all possible configurations in that grid search.
    n_repeats: int
        Number of experiments to run (with different random seeds) for each generated configuration.
    do_local_test: bool
        If True, sample one of the generated configurations and use it to run a short test locally,
        to ensure that the jobs will run properly.
    kind: str
        One of pbs, slurm, slurm-local, parallel, local. Specifies which method should be
        used to run the jobs in parallel.
    readme: str
        A string outlining the purpose/context for the created search.
    **run_kwargs:
        Additional arguments that are ultimately passed to `ParallelSession` in order to run the job.

    """
    with config:
        cfg.update_from_command_line()

    # Get run_kwargs from command line
    sig = inspect.signature(ParallelSession.__init__)
    default_run_kwargs = sig.bind_partial()
    default_run_kwargs.apply_defaults()
    cl_run_kwargs = clify.command_line(default_run_kwargs.arguments).parse()
    run_kwargs.update(cl_run_kwargs)

    if config.seed is None or config.seed < 0:
        config.seed = gen_seed()

    assert kind in "pbs slurm slurm-local parallel local".split()
    assert 'build_command' not in config
    config['build_command'] = ' '.join(sys.argv)
    print(config['build_command'])

    if kind == "local":
        with config:
            cfg.update_from_command_line()
            from dps.train import training_loop
            return training_loop()
    else:
        config.name = name

        config = config.copy(
            start_tensorboard=False,
            save_summaries=False,
            update_latest=False,
            show_plots=False,
            slim=False,
            max_experiments=np.inf,
        )

        if readme == "_vim_":
            readme = edit_text(
                prefix="dps_readme_", editor="vim", initial_text="README.md: \n")

        job, archive_path = build_search(
            cfg.build_experiments_dir, name, distributions, config,
            add_date=1, _zip=True, do_local_test=do_local_test,
            n_param_settings=n_param_settings, n_repeats=n_repeats,
            readme=readme)

        run_kwargs.update(
            archive_path=archive_path, name=name, wall_time=wall_time,
            kind=kind, parallel_exe=cfg.parallel_exe)

        parallel_session = submit_job(**run_kwargs)

        os.remove(str(archive_path))
        shutil.rmtree(str(archive_path).split('.')[0])

        return parallel_session


def dps_submit_cl():
    clify.wrap_function(submit_job)()


def submit_job(
        archive_path, name, wall_time="1year", ppn=1, cpp=1, pmem=0,
        queue="", kind="local", gpu_set="", **run_kwargs):

    assert kind in "pbs slurm slurm-local parallel".split()

    if "slurm" in kind and not pmem:
        raise Exception("Must supply a value for pmem (per-process-memory in mb) when using SLURM")

    run_kwargs.update(wall_time=wall_time, ppn=ppn, cpp=cpp, kind=kind, gpu_set=gpu_set, pmem=pmem)

    session = ParallelSession(
        name, archive_path, 'map', cfg.run_experiments_dir, dry_run=False, redirect=True,
        env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'), **run_kwargs)

    if kind in "parallel slurm-local".split():
        session.run()
        return session

    job_dir = session.job_directory

    python_script = """#!{}
import dill
with open("./session.pkl", "rb") as f:
    session = dill.load(f)
session.run()
""".format(sys.executable)
    with open(os.path.join(job_dir, "run.py"), 'w') as f:
        f.write(python_script)

    with open(os.path.join(job_dir, "session.pkl"), 'wb') as f:
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
        resources = "--nodes={} --ntasks-per-node={} --cpus-per-task={} --time={}".format(
            session.n_nodes, session.ppn, cpp, wall_time_minutes)

        if pmem:
            resources = "{} --mem-per-cpu={}mb".format(resources, pmem)

        if gpu_set:
            n_gpus = len([int(i) for i in gpu_set.split(',')])
            resources = "{} --gres=gpu:{}".format(resources, n_gpus)

        project = "def-jpineau"
        # project = "rpp-bengioy"
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
    return session
