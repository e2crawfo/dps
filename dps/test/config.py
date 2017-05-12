import numpy as np
import tensorflow as tf

from dps.updater import DifferentiableUpdater
from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.policy import SoftmaxSelect, EpsilonGreedySelect, GumbelSoftmaxSelect
from dps.utils import Config, CompositeCell, MLP
from dps.experiments import (
    arithmetic, simple_addition, pointer_following,
    hard_addition, lifted_addition, translated_mnist)


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

    verbose = False


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
    test_time_explore = None


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


class ArithmeticConfig(DefaultConfig):
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
    trainer = arithmetic.ArithmeticTrainer()
    visualize = arithmetic.visualize


class SimpleAdditionConfig(DefaultConfig):
    T = 30
    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
    ]

    # curriculum = [
    #     dict(width=1, n_digits=10),
    #     dict(width=2, n_digits=10),
    #     dict(width=3, n_digits=10),
    #     dict(width=4, n_digits=10),
    #     dict(width=5, n_digits=10),
    #     dict(width=6, n_digits=10),
    #     dict(width=7, n_digits=10),
    #     dict(width=8, n_digits=10),
    #     dict(width=9, n_digits=10),
    # ]

    log_dir = '/tmp/dps/simple_addition/'

    trainer = simple_addition.SimpleAdditionTrainer()
    visualize = simple_addition.visualize


class PointerConfig(DefaultConfig):
    T = 30
    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
        dict(width=4, n_digits=10)]
    log_dir = '/tmp/dps/pointer/'

    trainer = pointer_following.PointerTrainer()
    visualize = pointer_following.visualize


class HardAdditionConfig(DefaultConfig):
    T = 30
    curriculum = [
        dict(height=2, width=3, n_digits=2),
        dict(height=2, width=4, n_digits=2),
        dict(height=2, width=5, n_digits=2)]
    log_dir = '/tmp/dps/hard_addition/'
    trainer = hard_addition.HardAdditionTrainer()
    visualize = hard_addition.visualize


class LiftedAdditionConfig(DefaultConfig):
    T = 30
    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
        dict(width=4, n_digits=10),
        dict(width=5, n_digits=10)]
    log_dir = '/tmp/dps/lifted_addition/'

    trainer = lifted_addition.LiftedAdditionTrainer()
    visualize = lifted_addition.visualize


class TranslatedMnistConfig(DefaultConfig):
    T = 10
    curriculum = [
        dict(W=28, N=28),
        # dict(W=50, N=14),
    ]
    inc_delta = 0.2
    inc_x = 0.2
    inc_y = 0.2
    inc_sigma = 0.2
    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=8),
                                        MLP(),
                                        n_actions))

    # controller_func = staticmethod(
    #     lambda n_actions: FeedforwardCell(MLP([100, 100]), n_actions))

    log_dir = '/tmp/dps/translated_mnist/'

    base_kwargs = dict(inc_delta=0.2, inc_sigma=0.2, inc_x=0.2, inc_y=0.2)

    trainer = translated_mnist.TranslatedMnistTrainer()
    visualize = translated_mnist.visualize


algorithms = dict(
    diff=DiffConfig(),
    reinforce=ReinforceConfig(),
    qlearning=QLearningConfig())


tasks = dict(
    arithmetic=ArithmeticConfig(),
    simple_addition=SimpleAdditionConfig(),
    pointer=PointerConfig(),
    hard_addition=HardAdditionConfig(),
    lifted_addition=LiftedAdditionConfig(),
    translated_mnist=TranslatedMnistConfig())


def apply_mode(cfg, mode):
    if mode == "debug":
        cfg.update(DebugConfig())
