import pytest

from dps.experiments import lifted_addition
from dps.test import config


class LiftedAdditionConfig(config.DefaultConfig):
    T = 30
    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
        dict(width=4, n_digits=10),
        dict(width=5, n_digits=10)]


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_lifted_addition(config_str, mode, max_steps):
    cfg = LiftedAdditionConfig()

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

    lifted_addition.train(log_dir='/tmp/dps/lifted_addition/', config=cfg, seed=10)


def test_visualize_lifted_addition():
    lifted_addition.visualize(LiftedAdditionConfig())
