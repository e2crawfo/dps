import copy
from scipy.stats import distributions as spdists
import numpy as np

from dps.utils import gen_seed
from dps.test.config import algorithms, tasks
from dps.parallel.base import Job
from spectral_dagger.utils.experiment import ExperimentStore


class TupleDist(object):
    def __init__(self, *dists):
        self.dists = dists

    def rvs(self, shape):
        stack = []
        for d in self.dists:
            if hasattr(d, 'rvs'):
                stack.append(d.rvs(shape))
            else:
                s = np.zeros(shape, dtype=np.object)
                s.fill(d)
                stack.append(s)
        outp = np.zeros_like(stack[0], dtype=np.object)
        if isinstance(shape, int):
            shape = [shape]
        for index in np.ndindex(*shape):
            outp[index] = tuple(s[index] for s in stack)
        return outp


class ChoiceDist(object):
    def __init__(self, choices):
        self.choices = choices

    def rvs(self, shape):
        return np.random.choice(self.choices, shape)


class SeedDist(object):
    def rvs(self, shape):
        return np.random.randint(np.iinfo(np.int32).max, size=shape)


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
    samples = {k: d.rvs(n) for k, d in distributions.items()}
    configs = []
    for i in range(n):
        _config = copy.copy(base_config)
        for k, s in samples.items():
            setattr(_config, k, s[i])
        _config.idx = i
        for r in range(repeats):
            config = copy.copy(_config)
            config.repeat = r
            config.seed = gen_seed()
            configs.append(config)

    return configs


def build_search(n, repeats, path, alg, task):
    config = tasks[task]
    config.update(algorithms[alg])

    distributions = dict()
    if alg == 'reinforce':
        pass
    elif alg == 'qlearning':
        distributions.update(
            lr_schedule=TupleDist(spdists.uniform(0, 0.1), 1000, 0.9, False),
            exploration_schedule=TupleDist(spdists.uniform(0, 0.5), 1000, 0.9, False),
            batch_size=ChoiceDist(10 * np.arange(1, 11)),
            replay_max_size=ChoiceDist(2 ** np.arange(6, 14)),
            double=ChoiceDist([0, 1])
        )
    elif alg == 'diff':
        pass
    else:
        raise NotImplementedError("Unknown algorithm: {}.".format(alg))

    configs = sample_configs(n, repeats, config, distributions)

    es = ExperimentStore(str(path), max_experiments=10, delete_old=1)
    exp_dir = es.new_experiment('{}_{}'.format(alg, task), use_time=1, force_fresh=1)
    print(str(config))

    print("Building parameter search at {}.".format(exp_dir.path))

    job = Job(exp_dir.path)
    summaries = job.map(config.trainer.train, configs)
    best = job.reduce(reduce_hyper_results, summaries)


def reduce_hyper_results(*results):
    # Only take settings which got to the latest stage of the curriculum
    deepest = max(r['n_stages'] for r in results)
    results = [r for r in results if r['n_stages'] == deepest]

    # Take the best of those that got to the latest stage.
    return min(results, key=lambda r: r['output'][-1][2])


def _build_search(args):
    build_search(args.n, args.repeats, args.path, args.alg, args.task)


def hyper_search_cl():
    from dps.parallel.base import parallel_cl
    cmd = (
        'build', 'Build a hyper-parameter search.', _build_search,
        ('n', dict(help="Number of parameter settings to try.", type=int)),
        ('repeats', dict(help="Number of repeats for each parameter setting.", type=int)),
        ('path', dict(help="Location to save the built job.", type=str)),
        ('alg', dict(help="Algorithm to use for learning.", type=str)),
        ('task', dict(help="Task to test on.", type=str))
    )
    parallel_cl('Build, run and view hyper-parameter searches.', [cmd])
