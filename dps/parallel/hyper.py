import copy
from scipy.stats import distributions as spdists
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

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


def reduce_hyper_results(*results):
    # Only take settings which got to the latest stage of the curriculum
    deepest = max(r['n_stages'] for r in results)
    results = [r for r in results if r['n_stages'] == deepest]

    fig, subplots = plt.subplots(deepest, 2)

    x_lim = [np.inf, -np.inf]
    y_lim = [np.inf, -np.inf]
    for stage in range(deepest):
        stage_results = [r for r in results if r['n_stages'] > stage]
        n_reached = len(stage_results)
        print("Num settings that reached stage {}: {}.".format(stage, n_reached))

        losses = [sr['output'][stage][2] for sr in stage_results]

        # Losses reached on current stage.
        axis = subplots[stage, 0]
        n, bins, patches = axis.hist(losses, 50, normed=1, facecolor='green', alpha=0.75)
        axis.set_title('Individual')

        y_lim[0] = min(y_lim[0], min(n))
        y_lim[1] = max(y_lim[1], max(n))
        x_lim[0] = min(x_lim[0], min(bins))
        x_lim[1] = max(x_lim[1], max(bins))

        grouped = defaultdict(list)
        for sr in stage_results:
            grouped[sr['config']['idx']].append(sr['output'][stage][2])

        mean_losses = [np.mean(v) for v in grouped.values()]

        # Mean losses reached on current stage.
        axis = subplots[stage, 1]
        n, bins, patches = axis.hist(mean_losses, 50, normed=1, facecolor='blue', alpha=0.75)
        axis.set_title('Group Averages')

        y_lim[0] = min(y_lim[0], min(n))
        y_lim[1] = max(y_lim[1], max(n))
        x_lim[0] = min(x_lim[0], min(bins))
        x_lim[1] = max(x_lim[1], max(bins))

    x_diff = (x_lim[1] - x_lim[0])
    x_lim = (x_lim[0] - 0.05 * x_diff, x_lim[1] + 0.05 * x_diff)
    y_diff = (y_lim[1] - y_lim[0])
    y_lim = (y_lim[0] - 0.05 * y_diff, y_lim[1] + 0.05 * y_diff)

    for ax in subplots.flatten():
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    # Rank settings by performance on the greatest curriculum stage reached by any setting.
    reached_deepest = [r for r in results if r['n_stages'] == deepest]
    ranked = sorted(reached_deepest, key=lambda r: r['output'][-1][2])
    ranked = [(r['output'][-1][2], r['config']) for r in ranked]

    print("Best 3 ranked: ")
    print(ranked[:3])

    grouped = defaultdict(list)
    configs = {}
    for sr in reached_deepest:
        idx = sr['config']['idx']
        grouped[idx].append(sr['output'][-1][2])
        if idx not in configs:
            configs[idx] = sr['config']

    scores = {k: np.mean(v) for k, v in grouped.items()}
    mean_ranked = sorted(scores, key=lambda k: scores[k])
    mean_ranked = [(scores[idx], configs[idx]) for idx in mean_ranked]

    print("Best 3 group ranked: ")
    print(mean_ranked[:3])

    plt.tight_layout()

    plt.show()


def build_search(name, n, repeats, path, alg, task, _zip):
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
    exp_dir = es.new_experiment('{}_{}_{}'.format(name, alg, task), use_time=0, force_fresh=1)
    print(str(config))

    print("Building parameter search at {}.".format(exp_dir.path))

    job = Job(exp_dir.path)
    summaries = job.map(config.trainer.train, configs)
    best = job.reduce(reduce_hyper_results, summaries)

    if _zip:
        job.zip()


def _build_search(args):
    build_search(args.n, args.repeats, args.path, args.alg, args.task, args.zip)


def _zip_search(args):
    job = Job(args.path)
    archive_name = args.name or Path(args.path).stem
    job.zip(archive_name)


def hyper_search_cl():
    from dps.parallel.base import parallel_cl
    build_cmd = (
        'build', 'Build a hyper-parameter search.', _build_search,
        ('path', dict(help="Location to save the built job.", type=str)),
        ('name', dict(help="Memorable name for the search.", type=int)),
        ('n', dict(help="Number of parameter settings to try.", type=int)),
        ('repeats', dict(help="Number of repeats for each parameter setting.", type=int)),
        ('alg', dict(help="Algorithm to use for learning.", type=str)),
        ('task', dict(help="Task to test on.", type=str)),
        ('--zip', dict(action='store_true', help="Whether to compress the result."))
    )

    zip_cmd = (
        'zip', 'Zip up a job.', _zip_search,
        ('to_zip', dict(help="Path to the job we want to zip.", type=str)),
        ('name', dict(help="Optional path where archive should be created.", type=str, default='')),
    )

    parallel_cl('Build, run and view hyper-parameter searches.', [build_cmd, zip_cmd])
