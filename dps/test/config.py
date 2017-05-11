import numpy as np
import tensorflow as tf

from dps.updater import DifferentiableUpdater
from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.policy import SoftmaxSelect, EpsilonGreedySelect, GumbelSoftmaxSelect
from dps.utils import Config, CompositeCell, MLP


class DefaultConfig(Config):
    seed = 12

    optimizer_class = tf.train.RMSPropOptimizer
    updater_class = None

    power_through = True  # Whether to complete the entire curriculum, even if threshold not reached.
    max_steps = 100
    batch_size = 100
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-2
    patience = np.inf

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=32),
                                        MLP(),
                                        n_actions))

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.001, 1000, 0.96, False)
    noise_schedule = (0.0, 1000, 0.96, False)
    exploration_schedule = (10.0, 1000, 0.96, False)

    test_time_explore = None

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


class DiffConfig(Config):
    updater_class = DifferentiableUpdater
    test_time_explore = None
    # action_selection = SoftmaxSelect()
    action_selection = GumbelSoftmaxSelect(hard=0)
    noise_schedule = (0.1, 1000, 0.96, False)
    exploration_schedule = (1.0, 1000, 0.9, False)
    max_grad_norm = 1.0
    patience = np.inf
    T = 10


class ReinforceConfig(Config):
    updater_class = REINFORCE
    threshold = 1e-2
    action_selection = SoftmaxSelect()
    test_time_explore = None
    exploration_schedule = (10.0, 1000, 0.9, False)
    noise_schedule = (0.0, 1000, 0.96, False)
    lr_schedule = (0.01, 1000, 0.98, False)
    patience = np.inf
    max_grad_norm = 0.0
    gamma = 1.0


class QLearningConfig(Config):
    updater_class = QLearning
    threshold = 1e-2
    action_selection = EpsilonGreedySelect()
    exploration_schedule = (0.5, 1000, 0.9, False)
    lr_schedule = (0.001, 1000, 1.0, False)
    double = False
    replay_max_size = 100
    target_update_rate = 0.001
    recurrent = True
    patience = np.inf
    batch_size = 100
    test_time_explore = 0.0


class RealConfig(Config):
    max_steps = 10000


class DebugConfig(Config):
    max_steps = 4
    batch_size = 2
    n_train = 10
    n_val = 10
    n_test = 0

    display_step = 1
    eval_step = 1

    debug = True


def apply_mode(cfg, mode):
    if mode == "debug":
        cfg.update(DebugConfig())
    elif mode == "real":
        cfg.update(RealConfig())
