import argparse

import clify

from dps.test.config import algorithms, tasks


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('alg')
    parser.add_argument('task')
    args, _ = parser.parse_known_args()

    task = [t for t in tasks if t.startswith(args.task)]
    assert len(task) == 1, "Ambiguity in task selection, possibilities are: {}.".format(task)
    task = task[0]

    _algorithms = list(algorithms) + ['visualize']
    alg = [a for a in _algorithms if a.startswith(args.alg)]
    assert len(alg) == 1, "Ambiguity in alg selection, possibilities are: {}.".format(alg)
    alg = alg[0]

    config = tasks[task]
    if alg == 'visualize':
        config.display = True
        config.save_display = True
        config = clify.wrap_object(config).parse()
        config.visualize()
    else:
        config.update(algorithms[alg])
        config = clify.wrap_object(config).parse()
        config.trainer.train(config=config)


if __name__ == "__main__":
    run()
