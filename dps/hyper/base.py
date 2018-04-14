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
from collections import namedtuple, defaultdict
from tabulate import tabulate
from pprint import pprint, pformat

import clify

import dps
from dps import cfg
from dps.config import DEFAULT_CONFIG
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

    def extract_stage_data(self, fields=None, bare=False):
        """ Extract stage-by-stage data about the training runs.

        Parameters
        ----------
        bare: boolean
            If True, only returns the data. Otherwise, additionally returns the stage-by-stage config and meta-data.

        Returns
        -------
        A nested data structure containing the requested data.

        {param-setting-key: {(repeat, seed): (pd.DataFrame(), [dict()], dict())}}

        """
        stage_data = defaultdict(dict)
        if isinstance(fields, str):
            fields = fields.split()

        config_keys = list(set(self.dist_keys() + ['idx']))

        KeyTuple = namedtuple(self.__class__.__name__ + "Key", config_keys)

        for exp_path in self.experiment_paths:
            exp_data = FrozenTrainingLoopData(exp_path)

            md = {}
            md['host'] = exp_data.host
            for k in config_keys:
                md[k] = exp_data.config[k]

            sc = []
            records = []
            for stage in exp_data.history:
                record = stage.copy()

                if 'best_path' in record:
                    del record['best_path']

                sc.append(record['stage_config'])
                del record['stage_config']

                # Fix and filter keys
                _record = {}
                for k, v in record.items():
                    if k.startswith("best_"):
                        k = k[5:]

                    if (fields and k in fields) or not fields:
                        _record[k] = v

                records.append(_record)

            key = KeyTuple(*(exp_data.config[k] for k in config_keys))

            repeat = exp_data.config["repeat"]
            seed = exp_data.config["seed"]

            if bare:
                stage_data[key][(repeat, seed)] = pd.DataFrame.from_records(records)
            else:
                stage_data[key][(repeat, seed)] = (pd.DataFrame.from_records(records), sc, md)

        return stage_data

    def extract_step_data(self, mode, fields=None, stage=None):
        """ Extract per-step data across all experiments.

        Parameters
        ----------
        mode: str
            Data-collection mode to extract data from.
        fields: str
            Names of fields to extract data for. If not supplied, data for all
            fields is returned.
        stage: int or slice or tuple
            Specification of the stages to collect data for. If not supplied, data
            from all stages is returned.

        Returns
        -------

        A nested data structure containing the requested data.

        {param-setting-key: {(repeat, seed): pd.DataFrame()}}

        """
        step_data = defaultdict(dict)
        if isinstance(fields, str):
            fields = fields.split()

        config_keys = list(set(self.dist_keys() + ['idx']))

        KeyTuple = namedtuple(self.__class__.__name__ + "Key", config_keys)

        for exp_path in self.experiment_paths:
            exp_data = FrozenTrainingLoopData(exp_path)

            _step_data = exp_data.step_data(mode, stage)

            if fields:
                try:
                    _step_data = _step_data[fields]
                except KeyError:
                    print("Valid keys are: {}".format(_step_data.keys()))
                    raise

            key = KeyTuple(*(exp_data.config[k] for k in config_keys))

            repeat = exp_data.config["repeat"]
            seed = exp_data.config["seed"]

            step_data[key][(repeat, seed)] = _step_data

        return step_data

    def print_summary(self, print_config=True, verbose=False, criteria=None, maximize=False):
        """ Get all completed ops, get their outputs. Summarize em. """

        print("Summarizing search stored at {}.".format(os.path.realpath(self.path)))

        criteria_key = criteria if criteria else "stopping_criteria"

        keys = self.dist_keys()
        stage_data = self.extract_stage_data()

        best = []

        # For each parameter setting, identify the stage where it got the lowest value for `criteria_key`.
        for i, (key, value) in enumerate(sorted(stage_data.items())):
            _best = []

            for (repeat, seed), (df, sc, md) in value.items():
                if maximize:
                    _best.append(dict(df.iloc[df[criteria_key].idxmax()]))
                else:
                    _best.append(dict(df.iloc[df[criteria_key].idxmin()]))

                for key in keys:
                    _best[-1][key] = md[key]

            _best = pd.DataFrame.from_records(_best)
            _best = _best.sort_values(criteria_key)
            sc = _best[criteria_key].mean()
            best.append((sc, _best))

        best = sorted(best, reverse=not maximize, key=lambda x: x[0])
        best = [df for _, df in best]

        _column_order = [criteria_key, 'seed', 'reason', 'n_steps', 'host']
        column_order = [c for c in _column_order if c in best[0]]
        remaining = [k for k in best[0].keys() if k not in column_order and k not in keys]
        column_order = column_order + sorted(remaining)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('\n' + '*' * 100)
            direction = "DECREASING" if not maximize else "INCREASING"
            print("RESULTS GROUPED BY PARAM VALUES, ORDER OF {} VALUE OF <{}>: ".format(direction, criteria_key))

            for i, b in enumerate(best):
                print('\n {} '.format(len(best)-i) + '*' * 40)
                pprint({k: b[k].iloc[0] for k in keys})
                b = b.drop(keys, axis=1)[column_order]
                with_stats = pd.merge(
                    b.transpose(), b.describe().transpose(),
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
            pprint(self.dist())

        print(self.job.summary(verbose=verbose))


def nested_sample(param_dist, n_samples=1):
    """ Generate all samples from a distribution.

    Distribution must be specified as a dictionary mapping
    from names to either a list of possible values or a distribution
    (i.e. has a method `rvs`).

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

    def __call__(self, new):
        os.nice(10)

        start_time = time.time()

        print("Starting new training run at: ")
        print(datetime.datetime.now())

        print("Sampled values: ")
        print(new)

        print("Base config: ")
        print(self.base_config)

        exp_name = '_'.join("{}={}".format(k, new[k]) for k in 'idx repeat'.split())

        dps.reset_config()

        config = DEFAULT_CONFIG.copy()
        config.update(self.base_config)
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
    es = ExperimentStore(path, max_experiments=None, prefix="build_search")

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

    with open(exp_dir.path_for("sampled_configs.txt"), "w") as f:
        f.write("\n".join("idx={}: {}".format(config["idx"], pformat(config)) for config in new_configs))

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
