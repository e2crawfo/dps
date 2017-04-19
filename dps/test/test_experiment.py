import pytest
from dps.experiments.simple_addition import train_addition
from dps.experiments.pointer_following import train_pointer


@pytest.mark.parametrize('config', ['default', 'rl'])
def test_simple_addition(config):
    train_addition(log_dir='/tmp/dps/addition/', config=config)


def test_pointer_following():
    train_pointer(log_dir='/tmp/dps/pointer/', config='default')


def test_curriculum():
    train_pointer(log_dir='/tmp/dps/pointer/', config='curriculum')
