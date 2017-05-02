import pytest

from dps.experiments import hard_addition
from dps.test import config


class HardAdditionConfig(config.DefaultConfig):
    T = 30
    curriculum = [
        dict(height=2, width=2, n_digits=10),
        dict(height=2, width=3, n_digits=10),
        dict(height=2, width=4, n_digits=10),
        dict(height=2, width=5, n_digits=10)]


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_hard_addition(config_str, mode):
    cfg = HardAdditionConfig()

    if config_str == 'reinforce':
        cfg.update(config.ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(config.QLearningConfig())
    elif config_str == 'diff':
        cfg.update(config.DiffConfig())
    else:
        raise NotImplementedError()

    config.apply_mode(cfg, mode)

    hard_addition.train(log_dir='/tmp/dps/hard_addition/', config=cfg, seed=10)


def test_visualize_addition():
    hard_addition.visualize(HardAdditionConfig())
