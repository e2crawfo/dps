import pytest

from dps.test.config import algorithms, tasks


@pytest.mark.parametrize('task', sorted(tasks.keys()))
@pytest.mark.parametrize('alg', sorted(algorithms.keys()))
def test_production_system(task, alg, max_steps, verbose, display):

    config = tasks[task]
    config.update(algorithms[alg])

    config.verbose = verbose
    config.display = display

    if max_steps is not None:
        config.max_steps = int(max_steps)

    config.trainer.train(config=config, seed=100)


@pytest.mark.parametrize('task', sorted(tasks.keys()))
def test_visualize(task):
    config = tasks[task]
    config.display = True
    config.save_display = True
    config.visualize()
