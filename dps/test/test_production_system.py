import pytest

from dps.config import actor_configs, critic_configs, tasks, test_configs
from dps.train import training_loop
from dps.run import build_and_visualize


slow = pytest.mark.skipif(
    not pytest.config.getoption("--run-slow"),
    reason="need --run-slow option to run"
)


@pytest.mark.parametrize('task', sorted(tasks.keys()))
@pytest.mark.parametrize('actor', sorted(actor_configs.keys()))
@pytest.mark.parametrize('critic', sorted(critic_configs.keys()))
@slow
def test_train(task, actor, critic, max_steps, verbose, display):
    config = tasks[task]
    config.update(actor_configs[actor])
    config.update(critic_configs[critic])
    config.update(verbose=verbose, display=display, seed=100)

    if max_steps is not None:
        config.max_steps = int(max_steps)

    with config:
        val = training_loop()
        print(val)


@pytest.mark.parametrize('task', sorted(test_configs.keys()))
def test_visualize(task, display):
    config = test_configs[task]
    config.update(display=display, save_display=False)
    with config:
        build_and_visualize()
