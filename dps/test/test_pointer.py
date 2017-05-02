import pytest

from dps.experiments import pointer_following
from dps.test import config


class PointerConfig(config.DefaultConfig):
    T = 30
    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
        dict(width=4, n_digits=10)]


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_pointer_following(config_str, mode):
    cfg = PointerConfig()

    if config_str == 'reinforce':
        cfg.update(config.ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(config.QLearningConfig())
    elif config_str == 'diff':
        cfg.update(config.DiffConfig())
    else:
        raise NotImplementedError()

    config.apply_mode(cfg, mode)

    pointer_following.train(log_dir='/tmp/dps/pointer/', config=cfg, seed=10)


def test_visualize_pointer():
    pointer_following.visualize(PointerConfig())
