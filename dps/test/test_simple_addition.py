import pytest

from dps.experiments import simple_addition
from dps.test import config


class SimpleAdditionConfig(config.DefaultConfig):
    T = 30

    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
        dict(width=4, n_digits=10),
        dict(width=5, n_digits=10),
        dict(width=6, n_digits=10),
        dict(width=7, n_digits=10),
        dict(width=8, n_digits=10),
        dict(width=9, n_digits=10),
    ]

    log_dir = '/tmp/dps/simple_addition/'


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_simple_addition(config_str, mode, max_steps):
    cfg = SimpleAdditionConfig()

    if config_str == 'reinforce':
        cfg.update(config.ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(config.QLearningConfig())
    elif config_str == 'diff':
        cfg.update(config.DiffConfig())
    else:
        raise NotImplementedError()

    config.apply_mode(cfg, mode)
    if max_steps is not None:
        cfg.max_steps = int(max_steps)

    simple_addition.SimpleAdditionTrainer().train(config=cfg, seed=100)


def test_visualize_simple_addition():
    simple_addition.visualize(SimpleAdditionConfig())
