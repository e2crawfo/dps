import copy
import numpy as np
from pathlib import Path
import clify
import pandas as pd
from pprint import pprint
import time
import datetime
from collections import Sequence

from dps import cfg
from dps.train import training_loop
from dps.utils import gen_seed, Config
from dps.config import parse_task_actor_critic, tasks, actor_configs, critic_configs
from dps.parallel.base import Job, ReadOnlyJob
from spectral_dagger.utils.experiment import ExperimentStore


class ChoiceDist(object):
    def __init__(self, choices, p=None, dtype=None):
        self.choices = choices
        self.p = p
        if p is not None:
            assert len(p) == len(choices)
        self.dtype = dtype

    def rvs(self, shape=None):
        if shape is None:
            choice_idx = np.random.choice(len(self.choices), p=self.p)
            choice = self.choices[choice_idx]
            if hasattr(choice, 'rvs'):
                return choice.rvs()
            else:
                return choice

        indices = np.random.choice(len(self.choices), size=shape, p=self.p)
        results = np.zeros(shape, dtype=np.object)
        for i in range(len(self.choices)):
            equals_i = indices == i
            if hasattr(self.choices[i], 'rvs'):
                samples = self.choices[i].rvs(equals_i.sum())
                results[np.nonzero(equals_i)] = samples
            else:
                results[np.nonzero(equals_i)] = self.choices[i]

        if self.dtype:
            results = results.astype(self.dtype)
        elif hasattr(self.choices, 'dtype'):
            results = results.astype(self.choices.dtype)
        return results

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "ChoiceDist(\n    choices={},\n    p={},\n    dtype={})".format(
            self.choices, self.p, self.dtype)


class LogUniform(object):
    def __init__(self, lo, hi, base=None):
        self.lo = lo
        self.hi = hi
        self.base = base or np.e

    def rvs(self, shape=None):
        return self.base ** np.random.uniform(self.lo, self.hi, shape)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "LogUniform(lo={}, hi={}, base={})".format(self.lo, self.hi, self.base)


def nested_sample(d):
    if isinstance(d, dict):
        _d = d.copy()
        _d.update({k: nested_sample(v) for k, v in d.items()})
        return _d
    elif hasattr(d, 'rvs'):
        return d.rvs()
    else:
        return d


def nested_map(d, f):
    if isinstance(d, dict):
        _d = d.copy()
        _d.update({k: nested_map(v, f) for k, v in d.items()})
        return _d
    else:
        return f(d)


def sample_configs(n, repeats, base_config, distributions):
    """
    Parameters
    ----------
    n: int > 0
        Number of configs to sample.
    repeats: int > 0
        Number of different seeds to use for each sampled configuration.
    base_config: Config instance
        The base config, supplies any parameters not covered in ``distribution``.
    distributions: dict
        Mapping from parameter names to distributions (objects with
        member function ``rvs`` which accepts a shape and produces
        an array of samples with that shape).

    """
    distributions = nested_map(
        distributions,
        lambda e: ChoiceDist(list(e)) if isinstance(e, Sequence) else e)

    max_tries = 1000

    sample_traces = set()
    samples = []

    for i in range(n):
        n_tries = 0
        while True:
            sample = nested_sample(distributions)
            trace = str(sample)

            if trace not in sample_traces:
                break

            n_tries += 1

            if n_tries >= max_tries:
                raise Exception("Tried {} times, could not generate "
                                "a new unique configuration.".format(n_tries))

        sample_traces.add(trace)
        samples.append(sample)

    configs = []
    for s in samples:
        s['idx'] = i
        for r in range(repeats):
            _new = copy.deepcopy(s)
            _new['repeat'] = r
            _new['seed'] = gen_seed()
            configs.append(_new)

    return configs


def reduce_hyper_results(store, *results):
    distributions = store.load_object('metadata', 'distributions')
    distributions = Config(distributions)
    keys = list(distributions.leaf_keys())

    # Create a pandas dataframe storing the results
    records = []
    for r in results:
        record = dict(
            latest_stage=r['history'][-1]['stage'],
            total_steps=sum(s['n_steps'] for s in r['history']),
            final_stage_steps=r['history'][-1]['n_steps'],
            final_stage_loss=r['history'][-1]['best_loss'],
            final_stage_last_imp_step=r['history'][-1]['best_local_step'],
            reason=r['history'][-1]['reason'],
            host=r['host'])

        for k in keys:
            record[k] = r['config'][k]
        record['seed'] = r['config']['seed']
        records.append(record)
    df = pd.DataFrame.from_records(records)

    groups = df.groupby(keys)

    data = []
    for k, _df in groups:
        _df = _df.sort_values(['latest_stage', 'final_stage_loss'])
        data.append(dict(
            data=_df,
            keys=k,
            latest_stage=_df.latest_stage.max(),
            stage_sum=_df.latest_stage.sum(),
            final_stage_loss=_df.final_stage_loss.mean()))

    data = sorted(data, reverse=False, key=lambda x: (x['latest_stage'], -x['final_stage_loss'], x['stage_sum']))

    column_order = [
        'latest_stage', 'final_stage_loss', 'seed', 'reason', 'total_steps',
        'final_stage_steps', 'final_stage_last_imp_step', 'host']

    print('\n' + '*' * 100)
    print("RESULTS GROUPED BY PARAM VALUES, WORST COMES FIRST: ")
    for i, d in enumerate(data):
        print('\n {} '.format(len(data)-i) + '*' * 40)
        pprint({n: v for n, v in zip(keys, d['keys'])})
        _data = d['data'].drop(keys, axis=1)
        _data = _data[column_order]
        print(_data)

    print('\n' + '*' * 100)
    print("BASE CONFIG")
    print(store.load_object('metadata', 'config'))

    print('\n' + '*' * 100)
    print("DISTRIBUTIONS")
    pprint(distributions)


class RunTrainingLoop(object):
    def __init__(self, base_config):
        self.base_config = base_config

    def __call__(self, new):
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

            val = training_loop(start_time=start_time)

        return val


def build_search(path, name, n, repeats, distributions, _zip, config=None, use_time=0):
    """ Create a Job representing a hyper-parameter search.

    Parameters
    ----------
    path: str
        Path to the directory where the archive that is built for the search will be saved.
    name: str
        Name for the search.
    n: int
        Number of parameter settings to sample.
    repeats: int
        Number of different random seeds to run each sample with.
    distributions: dict (str -> distribution)
        Distributions to sample from.
    _zip: bool
        Whether to zip the created search directory.
    config: Config instance
        The base configuration.
    use_time: bool
        Whether to add time to name of experiment directory.

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

    new_configs = sample_configs(n, repeats, config, distributions)
    job.map(RunTrainingLoop(config), new_configs)

    job.save_object('metadata', 'distributions', distributions)
    job.save_object('metadata', 'config', config)

    if _zip:
        path = job.zip(delete=False)
    else:
        path = exp_dir.path

    return job, path


def _build_search(args):
    return build_search(args.path, args.name, args.n, args.repeats, args.alg, args.task, not args.no_zip)


def _summarize_search(args):
    # Get all completed jobs, get their outputs. Plot em.
    job = ReadOnlyJob(args.path)
    results = [op.get_outputs(job.objects)[0] for op in job.completed_ops() if 'map' in op.name]
    reduce_hyper_results(job.objects, *results)


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
