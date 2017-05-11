import pytest
import tensorflow as tf

from dps.experiments import simple_arithmetic
from dps.test import config
from dps.utils import CompositeCell, MLP


class ArithmeticConfig(config.DefaultConfig):
    T = 3
    curriculum = [
        dict(order=[0], T=1),
        dict(order=[0, 1], T=2),
        dict(order=[0, 1, 0], T=3)]
    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=256),
                                        MLP(),
                                        n_actions))
    log_dir = '/tmp/dps/arithmetic/'


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_simple_arithmetic(config_str, mode, max_steps):
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
    if max_steps is not None:
        cfg.max_steps = int(max_steps)

    simple_arithmetic.ArithmeticTrainer().train(cfg, seed=20)


def test_visualize_arithmetic():
    simple_arithmetic.visualize(ArithmeticConfig())
