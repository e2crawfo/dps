import pytest

from dps.experiments import simple_arithmetic
from dps.test import config


class ArithmeticConfig(config.DefaultConfig):
    T = 3
    curriculum = [
        dict(order=[0], T=1),
        dict(order=[0, 1], T=2),
        dict(order=[0, 1, 0], T=3)]


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_simple_arithmetic(config_str, mode):
    cfg = ArithmeticConfig()

    if config_str == 'reinforce':
        cfg.update(config.ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(config.QLearningConfig())
    elif config_str == 'diff':
        cfg.update(config.DiffConfig())
    else:
        raise NotImplementedError()

    config.apply_mode(cfg, mode)

    simple_arithmetic.train(log_dir='/tmp/dps/arithmetic/', config=cfg, seed=20)


def test_visualize_arithmetic():
    simple_arithmetic.visualize(ArithmeticConfig())
