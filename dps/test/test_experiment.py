import pytest
from dps.experiments.simple_addition import train_addition
from dps.experiments.pointer_following import train_pointer, visualize


@pytest.mark.parametrize('config', ['default', 'rl', 'curriculum', 'rlcurriculum'])
def test_simple_addition(config):
    train_addition(log_dir='/tmp/dps/addition/', config=config, seed=20)


@pytest.mark.parametrize('config', ['default', 'rl', 'curriculum'])
def test_pointer_following(config):
    train_pointer(log_dir='/tmp/dps/pointer/', config=config, seed=10)


def test_build_and_visualize():
    visualize()
