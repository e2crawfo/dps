import pytest

from dps.test.config import algorithms, tasks, apply_mode


@pytest.mark.parametrize('task', sorted(tasks.keys()))
@pytest.mark.parametrize('alg', sorted(algorithms.keys()))
def test_production_system(task, alg, mode, max_steps):

    config = tasks[task]
    config.update(algorithms[alg])

    apply_mode(config, mode)
    if max_steps is not None:
        config.max_steps = int(max_steps)

    config.trainer.train(config=config, seed=100)


@pytest.mark.parametrize('task', sorted(tasks.keys()))
def test_visualize(task):
    config = tasks[task]
    config.visualize()
