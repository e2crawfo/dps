import argparse

import clify

from dps.test.config import algorithms, tasks, test_configs
from dps.production_system import build_and_visualize
from dps.utils import pdb_postmortem


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('alg')
    parser.add_argument('task')
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")
    args, _ = parser.parse_known_args()

    task = [t for t in tasks if t.startswith(args.task)]
    assert len(task) == 1, "Ambiguity in task selection, possibilities are: {}.".format(task)
    task = task[0]

    _algorithms = list(algorithms) + ['visualize']
    alg = [a for a in _algorithms if a.startswith(args.alg)]
    assert len(alg) == 1, "Ambiguity in alg selection, possibilities are: {}.".format(alg)
    alg = alg[0]

    if args.pdb:
        with pdb_postmortem():
            _run(alg, task)
    else:
        _run(alg, task)


def _run(alg, task, _config=None, load_from=None):
    if alg == 'visualize':
        config = test_configs[task]
        if _config is not None:
            config.update(_config)
        config = clify.wrap_object(config).parse()
        config.update(display=True, save_display=True)

        with config:
            build_and_visualize(load_from=load_from)
    else:
        config = tasks[task]
        config.update(algorithms[alg])
        if _config is not None:
            config.update(_config)

        config = clify.wrap_object(config).parse()

        with config:
            config.trainer.train()
