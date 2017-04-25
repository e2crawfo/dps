import pytest
import numpy as np

from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.policy import SoftmaxSelect, EpsilonGreedySelect
from dps.experiments import simple_addition
from dps.experiments import pointer_following


class ReinforceConfig(object):
    updater_class = REINFORCE
    threshold = 1e-2
    action_selection = SoftmaxSelect()


class QLearningConfig(object):
    updater_class = QLearning
    threshold = 1e-2
    action_selection = EpsilonGreedySelect()
    exploration_schedule = (0.05, 1000, 1.0, False)
    lr_schedule = (0.01, 1000, 0.8, False)
    double = False
    replay_max_size = 10000
    target_update_rate = 0.01
    recurrent = True
    patience = np.inf
    batch_size = 100


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_simple_addition(config_str):
    cfg = simple_addition.AdditionConfig()
    cfg.curriculum = [
        dict(order=[0], T=1),
        dict(order=[0, 1], T=2),
        dict(order=[0, 1, 0], T=3)]

    if config_str == 'reinforce':
        cfg.update(ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(QLearningConfig())
        cfg.curriculum = [dict(order=[0, 1], T=2)]
    elif config_str == 'diff':
        pass
    else:
        raise NotImplementedError()

    simple_addition.train(log_dir='/tmp/dps/addition/', config=cfg, seed=20)


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_pointer_following(config_str):
    cfg = pointer_following.PointerConfig()

    cfg.curriculum = [
        dict(width=1, n_digits=3),
        dict(width=2, n_digits=3)]

    if config_str == 'reinforce':
        cfg.update(ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(QLearningConfig())
        cfg.T = 4
    elif config_str == 'diff':
        pass
    else:
        raise NotImplementedError()
    pointer_following.train(log_dir='/tmp/dps/pointer/', config=cfg, seed=10)


def test_build_and_visualize():
    pointer_following.visualize()
