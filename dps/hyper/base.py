import copy
import numpy as np
import pandas as pd
import time
import datetime
from itertools import product
from copy import deepcopy
import os
import sys
import inspect
from collections import namedtuple
from tabulate import tabulate
from pprint import pprint

import clify

import dps
from dps import cfg
from dps.utils import gen_seed, Config, ExperimentStore, edit_text
from dps.parallel import Job, ReadOnlyJob
from dps.train import FrozenTrainingLoopData
from dps.hyper.submit_job import submit_job, ParallelSession


class HyperSearch(object):
    """ Interface to a directory storing a hyper-parameter search. """

    def __init__(self, path):
        self.path = path

        job_path = os.path.join(path, 'results.zip')
        if not os.path.exists(job_path):
            job_path = os.path.join(path, 'orig.zip')
            assert os.path.exists(job_path)

        self.job = ReadOnlyJob(job_path)

    @property
    def objects(self):
        return self.job.objects

    def dist_keys(self):
        """ The keys that were searched over. """
        distributions = self.objects.load_object('metadata', 'distributions')
        if isinstance(distributions, list):
            keys = set()
            for d in distributions:
                keys |= set(d.keys())
            keys = list(keys)
        else:
            distributions = Config(distributions)
            keys = list(distributions.keys())

        return sorted(keys)

    def dist(self):
        return self.objects.load_object('metadata', 'distributions')

    def sampled_configs(self):
        pass

    @property
    def experiment_paths(self):
        experiments_dir = os.path.join(self.path, 'experiments')
        exp_dirs = os.listdir(experiments_dir)
        return [os.path.join(experiments_dir, ed) for ed in exp_dirs]

    def extract_summary_data(self):
        """ Extract high-level data about the training runs. """

        config_keys = self.dist_keys() + 'seed repeat idx'.split()

        records = []
        for exp_path in self.experiment_paths:
            exp_data = FrozenTrainingLoopData(exp_path)
            record = exp_data.history[-1].copy()

            record['host'] = exp_data.host
            if 'best_path' in record:
                del record['best_path']

            for k in config_keys:
                record[k] = exp_data.config[k]

            total_steps = sum(h.get('n_steps', 0) for h in exp_data.history)

            record.update(total_steps=total_steps, latest_stage=exp_data.n_stages-1)

            records.append(record)

        df = pd.DataFrame.from_records(records)
        for key in config_keys:
            df[key] = df[key].fillna(-np.inf)

        return df

    def extract_step_data(self, mode, field=None, stage=None):
        """ Extract per-step data across all training runs.

        Parameters
        ----------
        mode: str
            Data-collection mode to extract data from. Must be one of
            `train`, `off_policy`, `val`, `test`.
        field: str
            Name of field to extract data for. If not supplied, data for all
            fields is returned.
        stage: int or slice or tuple
            Specification of the stages to collect data for. If not supplied, data
            from all stages is returned.

        """
        step_data = {}

        config_keys = self.dist_keys() + 'seed repeat idx'.split()

        KeyTuple = namedtuple(
            self.__class__.__name__ + "Key", config_keys)

        for exp_path in self.experiment_paths:
            exp_data = FrozenTrainingLoopData(exp_path)

            _step_data = exp_data.step_data(mode, stage)
            if field:
                _step_data = _step_data[field]

            key = KeyTuple(*(exp_data.config[k] for k in config_keys))
            step_data[key] = _step_data

        return step_data

    def print_summary(self, print_config=True, verbose=False):
        """ Get all completed ops, get their outputs. Summarize em. """

        print("Summarizing self stored at {}.".format(os.path.realpath(self.path)))

        dist = Config(self.dist())
        keys = self.dist_keys()
        df = self.extract_summary_data()

        groups = df.groupby(keys)

        data = []
        for k, _df in groups:
            _df = _df.sort_values(['latest_stage', 'best_stopping_criteria'])
            data.append(dict(
                data=_df,
                keys=[k] if len(dist) == 1 else k,
                latest_stage=_df.latest_stage.max(),
                stage_sum=_df.latest_stage.sum(),
                best_stopping_criteria=_df.best_stopping_criteria.mean()))

        data = sorted(data, reverse=False, key=lambda x: (x['latest_stage'], x['best_stopping_criteria'], x['stage_sum']))

        _column_order = ['latest_stage', 'best_stopping_criteria', 'seed', 'reason', 'total_steps', 'n_steps', 'host']

        column_order = [c for c in _column_order if c in data]
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

        if print_config:
            print('\n' + '*' * 100)
            print("BASE CONFIG")
            print(self.objects.load_object('metadata', 'config'))

            print('\n' + '*' * 100)
            print("PARAMETER DISTRIBUTION")
            pprint(dist)

        print(self.job.summary(verbose=verbose))


def nested_sample(param_dist, n_samples=1):
    """ Generate all samples from a distribution.

    Distribution must be specified as a dictionary mapping
    from names to lists of possible values or param_dist.

    """
    assert isinstance(param_dist, dict)
    config = Config(param_dist)
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
        configs.append(type(param_dist)(new))
    return configs


def generate_all(param_dist):
    """ Generate all samples from a parameter distribution.

    Distribution must be specified as a dictionary mapping
    from names to lists of possible values.

    """
    assert isinstance(param_dist, dict)
    config = Config(param_dist)
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
        configs.append(type(param_dist)(new))
    return configs


def sample_configs(distributions, n_repeats, n_samples=None):
    """ Samples configs from a distribution for hyper-parameter search.

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


class _RunTrainingLoop(object):
    """ Entry point for each process. """

    def __init__(self, base_config):
        self.base_config = base_config

    @staticmethod
    def _sanitize_string(s):
        return str(s).replace('_', '-').replace(' ', '').replace('/', ',')

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
            "{}={}".format(self._sanitize_string(k), self._sanitize_string(new[k]))
            for k in keys)

        dps.reset_config()

        config = copy.copy(self.base_config)
        config.update(new)

        with config:
            cfg.update_from_command_line()

            from dps.train import training_loop
            return training_loop(exp_name=exp_name, start_time=start_time)


def build_search(
        path, name, distributions, config, n_repeats, n_param_settings=None,
        _zip=True, add_date=0, do_local_test=True, readme=""):
    """ Create a job implementing a hyper-parameter search.

    Parameters
    ----------
    path: str
        Path to the directory where the search archive will be saved.
    name: str
        Name for the search.
    distributions: dict (str -> (list or distribution))
        Distributions to sample from. Can also be a list of samples.
    config: Config instance
        The base configuration.
    n_repeats: int
        Number of different random seeds to run each sample with.
    n_param_settings: int
        Number of parameter settings to sample. If not supplied, all
        possibilities are generated.
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
    es = ExperimentStore(path, max_experiments=None)

    count = 0
    base_name = name
    has_built = False
    while not has_built:
        try:
            exp_dir = es.new_experiment(
                name, config.seed, add_date=add_date, force_fresh=1)
            has_built = True
        except FileExistsError:
            name = "{}_{}".format(base_name, count)
            count += 1

    if readme:
        with open(exp_dir.path_for('README.md'), 'w') as f:
            f.write(readme)

    print(config)
    exp_dir.record_environment(config=config, git_modules=dps)

    print("Building parameter search at {}.".format(exp_dir.path))

    job = Job(exp_dir.path)

    new_configs = sample_configs(distributions, n_repeats, n_param_settings)

    print("{} configs were sampled for parameter search.".format(len(new_configs)))

    if do_local_test:
        print("\nStarting local test " + ("=" * 80))
        test_config = new_configs[0].copy()
        test_config.update(max_steps=1000, render_hook=None)
        _RunTrainingLoop(config)(test_config)
        print("Done local test " + ("=" * 80) + "\n")

    job.map(_RunTrainingLoop(config.copy()), new_configs)

    job.save_object('metadata', 'distributions', distributions)
    job.save_object('metadata', 'config', config)

    print(job.summary())

    if _zip:
        path = job.zip(delete=True)
    else:
        path = exp_dir.path

    print("Zipped {} as {}.".format(exp_dir.path, path))

    return path


def build_and_submit(
        name, config, distributions=None, wall_time="1year", n_param_settings=0,
        n_repeats=1, do_local_test=False, kind="local", readme="", **run_kwargs):
    """ Build a job and submit it. Meant to be called from within a script.

    Parameters
    ----------
    name: str
        Name of the experiment.
    config: Config instance or dict
        Configuration to use as the base config for all jobs.
    distributions: dict
        Object used to generate variations of the base config (so that different
        jobs test different parameters).
    wall_time: str
        Maximum amount of time that is to be used to complete jobs.
    n_param_settings: int
        Number of different configurations to sample from `distributions`. If not
        supplied, it is assumed that `distributions` actually specifies a grid
        search, and an attempt is made to generate all possible configurations int
        that grid search.
    n_repeats: int
        Number of experiments to run (with different random seeds) for each
        generated configuration.
    do_local_test: bool
        If True, sample one of the generated configurations and use it to run a
        short test locally, to ensure that the jobs will run properly.
    kind: str
        One of pbs, slurm, slurm-local, parallel, local. Specifies which method
        should be used to run the jobs in parallel.
    readme: str
        A string outlining the purpose/context for the created search.
    **run_kwargs:
        Additional arguments that are ultimately passed to `ParallelSession` in
        order to run the job.

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

        archive_path = build_search(
            cfg.build_experiments_dir, name, distributions, config,
            add_date=1, _zip=True, do_local_test=do_local_test,
            n_param_settings=n_param_settings, n_repeats=n_repeats, readme=readme)

        run_kwargs.update(
            archive_path=archive_path, name=name, wall_time=wall_time,
            kind=kind, parallel_exe=cfg.parallel_exe)

        parallel_session = submit_job(**run_kwargs)

        return parallel_session
