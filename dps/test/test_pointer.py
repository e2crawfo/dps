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
    log_dir = '/tmp/dps/pointer/'


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_pointer_following(config_str, mode, max_steps):
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
    if max_steps is not None:
        cfg.max_steps = int(max_steps)

    pointer_following.PointerTrainer().train(config=cfg, seed=10)


def test_visualize_pointer():
    pointer_following.visualize(PointerConfig())
