import pytest

from dps.test.config import algorithms, tasks, test_configs
from dps.production_system import build_and_visualize


slow = pytest.mark.skipif(
    not pytest.config.getoption("--run-slow"),
    reason="need --run-slow option to run"
)


@pytest.mark.parametrize('task', sorted(tasks.keys()))
@pytest.mark.parametrize('alg', sorted(algorithms.keys()))
@slow
def test_production_system(task, alg, max_steps, verbose, display):

    config = tasks[task]
    config.update(algorithms[alg])

    config.verbose = verbose
    config.display = display

    if max_steps is not None:
        config.max_steps = int(max_steps)

    config.trainer.train(config=config, seed=100)


@pytest.mark.parametrize('task', sorted(test_configs.keys()))
def test_visualize(task, display):
    config = test_configs[task]
    config.display = display
    config.save_display = False
    with config.as_default():
        build_and_visualize()
