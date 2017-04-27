import pytest
import numpy as np
import tensorflow as tf

from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.experiments import simple_arithmetic, pointer_following, addition
from dps.utils import Config, CompositeCell, MLP
from dps.policy import SoftmaxSelect, EpsilonGreedySelect, GumbelSoftmaxSelect
from dps.updater import DifferentiableUpdater


def pytest_addoption(parser):
    parser.addoption("--debug", action="store_true", help="Run small tests in debug mode.")


@pytest.fixture
def debug(request):
    return request.config.getoption("--debug")


class DebugConfig(object):
    max_steps = 4
    batch_size = 2
    n_train = 10
    n_val = 10
    n_test = 0

    display_step = 1
    eval_step = 1

    debug = True


class DiffConfig(object):
    test_time_explore = 0.01
    action_selection = staticmethod(GumbelSoftmaxSelect(hard=0))
    exploration_schedule = (10.0, 1000, 0.9, False)


class ReinforceConfig(object):
    updater_class = REINFORCE
    threshold = 1e-2
    action_selection = SoftmaxSelect()
    test_time_explore = 0.01
    exploration_schedule = (10.0, 1000, 0.9, False)


class QLearningConfig(object):
    updater_class = QLearning
    threshold = 1e-2
    action_selection = EpsilonGreedySelect()
    exploration_schedule = (1.0, 1000, 1.0, False)
    lr_schedule = (0.01, 1000, 1.0, False)
    double = False
    replay_max_size = 1000
    target_update_rate = 0.01
    recurrent = True
    patience = np.inf
    batch_size = 10
    test_time_explore = 0.0


class ArithmeticConfig(Config):
    seed = 10

    T = 3
    curriculum = [
        dict(order=[0], T=1),
        dict(order=[0, 1], T=2),
        dict(order=[0, 1, 0], T=3)]
    optimizer_class = tf.train.RMSPropOptimizer
    updater_class = DifferentiableUpdater

    max_steps = 10000
    batch_size = 100
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-2
    patience = 100

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=32),
        MLP(),
        simple_arithmetic.Arithmetic.n_actions)
    action_selection = staticmethod(GumbelSoftmaxSelect(hard=0))

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.96, False)
    noise_schedule = (0.0, 10, 0.96, False)
    exploration_schedule = (10.0, 100, 0.96, False)

    test_time_explore = None

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_simple_arithmetic(config_str, debug):
    cfg = ArithmeticConfig()

    if config_str == 'reinforce':
        cfg.update(ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(QLearningConfig())
    elif config_str == 'diff':
        pass
    else:
        raise NotImplementedError()

    if debug:
        cfg.update(DebugConfig())

    simple_arithmetic.train(log_dir='/tmp/dps/arithmetic/', config=cfg, seed=20)


class PointerConfig(Config):
    seed = 12

    T = 8
    curriculum = [
        dict(width=1, n_digits=3),
        dict(width=2, n_digits=3),
        dict(width=3, n_digits=3),
        dict(width=4, n_digits=3)]

    optimizer_class = tf.train.RMSPropOptimizer
    updater_class = DifferentiableUpdater

    max_steps = 10000
    batch_size = 10
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-2
    patience = np.inf

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=32),
        MLP(),
        pointer_following.Pointer.n_actions)

    action_selection = staticmethod(GumbelSoftmaxSelect(hard=0))

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.99, False)
    noise_schedule = (0.0, 1000, 0.96, False)
    exploration_schedule = (10.0, 1000, 0.96, False)

    test_time_explore = None

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_pointer_following(config_str, debug):
    cfg = PointerConfig()

    if config_str == 'reinforce':
        cfg.update(ReinforceConfig())
    elif config_str == 'qlearning':
        cfg.update(QLearningConfig())
        cfg.T = 4
    elif config_str == 'diff':
        cfg.update(DiffConfig())
        cfg.curriculum = [
            dict(width=1, n_digits=3, T=4),
            dict(width=2, n_digits=3, T=4),
            dict(width=3, n_digits=3, T=4),
            dict(width=4, n_digits=3, T=6)]
    else:
        raise NotImplementedError()

    if debug:
        cfg.update(DebugConfig())

    pointer_following.train(log_dir='/tmp/dps/pointer/', config=cfg, seed=10)


class AdditionConfig(Config):
    seed = 12

    T = 10
    curriculum = [dict(width=1, n_digits=10)]

    optimizer_class = tf.train.RMSPropOptimizer
    updater_class = DifferentiableUpdater

    max_steps = 10000
    batch_size = 10
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-2
    patience = np.inf

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=32),
        MLP(),
        addition.Addition.n_actions)

    action_selection = staticmethod(GumbelSoftmaxSelect(hard=0))

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.99, False)
    noise_schedule = (0.0, 1000, 0.96, False)
    exploration_schedule = (10.0, 1000, 0.96, False)

    test_time_explore = None

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


@pytest.mark.parametrize('config_str', ['diff', 'reinforce', 'qlearning'])
def test_addition(config_str, debug):
    cfg = AdditionConfig()

    if config_str == 'reinforce':
        cfg.update(ReinforceConfig())
        cfg.exploration_schedule = (10.0, 1000, 1.0, False)
    elif config_str == 'qlearning':
        cfg.update(QLearningConfig())
    elif config_str == 'diff':
        cfg.update(DiffConfig())
    else:
        raise NotImplementedError()

    if debug:
        cfg.update(DebugConfig())

    addition.train(log_dir='/tmp/dps/addition/', config=cfg, seed=10)


def test_visualize_pointer():
    pointer_following.visualize(PointerConfig())


def test_visualize_addition():
    addition.visualize(AdditionConfig())
