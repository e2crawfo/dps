import pytest

import tensorflow as tf

from dps.experiments import translated_mnist
from dps.test import config
from dps.utils import CompositeCell, MLP


class TranslatedMnistConfig(config.DefaultConfig):
    T = 10
    curriculum = [
        dict(W=35, N=14),
        dict(W=50, N=14),
    ]
    inc_delta = 0.2
    inc_x = 0.2
    inc_y = 0.2
    inc_sigma = 0.2
    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=256),
                                        MLP(),
                                        n_actions))


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_translated_mnist(config_str, mode, max_steps):
    cfg = TranslatedMnistConfig()

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

    translated_mnist.train(log_dir='/tmp/dps/translated_mnist/', config=cfg, seed=10)


def test_visualize_translated_mnist():
    translated_mnist.visualize(TranslatedMnistConfig())
