import numpy as np
import tensorflow as tf
import os

from dps.updater import DifferentiableUpdater
from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.policy import SoftmaxSelect, EpsilonGreedySelect, GumbelSoftmaxSelect, IdentitySelect
from dps.utils import Config, CompositeCell, FeedforwardCell, MLP, DpsConfig, FixedController
from dps.experiments import (
    hello_world, simple_addition, pointer_following,
    hard_addition, translated_mnist, mnist_arithmetic, simple_arithmetic)


class DefaultConfig(DpsConfig):
    seed = 12

    preserve_policy = True  # Whether to use the policy learned on the last stage of the curriculum for each new stage.

    optimizer_spec = "rmsprop"
    updater_class = None

    power_through = True  # Whether to complete the entire curriculum, even if threshold not reached.
    max_steps = 100
    batch_size = 100
    n_train = 10000
    n_val = 200
    n_test = 0

    threshold = 1e-2
    patience = np.inf

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000
    n_controller_units = 32

    controller_func = lambda self, n_actions: CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=self.n_controller_units), MLP(), n_actions)

    lr_schedule = "exponential 0.001 1000 0.96"
    exploration_schedule = "exponential 10.0 1000 0.96"
    noise_schedule = None

    test_time_explore = None

    max_grad_norm = 0.0
    l2_norm_penalty = 0.0
    gamma = 1.0
    reward_window = 0.1

    n_auxiliary_tasks = 0
    auxiliary_coef = 0

    debug = False
    verbose = False
    display = False
    save_display = False
    path = os.getcwd()
    max_time = 0


class DiffConfig(Config):
    updater_class = DifferentiableUpdater
    test_time_explore = None
    # action_selection = SoftmaxSelect()
    action_selection = GumbelSoftmaxSelect(hard=0)
    max_grad_norm = 1.0
    patience = np.inf
    T = 10

    noise_schedule = "exponential 0.1 1000 0.96"


class ReinforceConfig(Config):
    updater_class = REINFORCE
    action_selection = SoftmaxSelect()
    test_time_explore = 0.1
    patience = np.inf

    entropy_schedule = "exp 0.1 100000 1.0"
    lr_schedule = "exp 0.001 1000 1.0"
    exploration_schedule = "exp 10.0 100000 1.0"
    # exploration_schedule = "poly 10.0 100000 1.0 1.0"
    gamma = 0.99


class QLearningConfig(Config):
    updater_class = QLearning
    action_selection = EpsilonGreedySelect()
    double = False

    lr_schedule = "exponential 0.00025 1000 1.0"
    exploration_schedule = "polynomial 1.0 10000 0.1 1"

    optimizer_spec = "rmsprop"

    replay_max_size = 1000000
    replay_threshold = -0.5
    replay_proportion = 0.0
    gamma = 0.99

    target_update_rate = None
    steps_per_target_update = 10000
    recurrent = True
    patience = np.inf
    samples_per_update = 32  # Number of rollouts between parameter updates
    update_batch_size = 32  # Number of sample rollouts to use for each parameter update
    batch_size = 64  # Number of sample experiences to execute
    test_time_explore = 0.05

    l2_norm_penalty = 0.0
    max_grad_norm = 0.0

    n_controller_units = 256


class DQNConfig(QLearningConfig):
    # From Nature paper

    # Rewards are clipped: all negative rewards set to -1, all positive set to 1, 0 unchanged.
    batch_size = 32

    lr_schedule = "0.00025"

    # annealed linearly from 1 to 0.1 over first million frames,
    # fixed at 0.1 thereafter "
    exploration_schedule = "polynomial 1.0 10000 0.1 1"

    replay_max_size = 1e6

    # max number of frames/states: 10 million

    test_time_explore = 0.05
    # target_network_update Once every 10000 frames
    steps_per_target_update = 10000

    gamma = 0.99

    # 4 actions selected between each update
    # RMS prop momentum: 0.95
    # squared RMS prop gradient momentum: 0.95
    # min squared gradient (RMSProp): 0.01
    # exploration 1 to 0.1 over 1,000,000 frames
    # total number of frames: 50,000,000, but because of frame skip, equivalent to 200,000,000 frames


class FpsConfig(QLearningConfig):
    gamma = 0.99
    update_batch_size = 32
    batch_size = 64

    # Exploration: annealed linearly from 1 to 0.1 over first million steps, fixed at 0.1 thereafter

    # Replay max size: 1million *frames*

    # They actually update every 4 *steps*, rather than every 4 experiences
    samples_per_update = 4


class DuelingConfig(QLearningConfig):
    max_grad_norm = 10.0
    lr_schedule = "6.25e-5"  # when prioritized experience replay was used
    test_time_explore = 0.001 # fixed at this value...this might also be the training exploration, its not clear


class DoubleConfig(QLearningConfig):
    double = True
    exploration_start = 0.1 # Not totally clear, but seems like they use the same scheme as DQN, but go from 1 to 0.01, instead of 1 to 0.1
    test_time_explore = 0.001
    # Target network update rate: once every 30,000 frames (DQN apparently does it once every 10,000 frames).


class DebugConfig(Config):
    max_steps = 4
    batch_size = 2
    n_train = 10
    n_val = 10
    n_test = 0

    display_step = 1
    eval_step = 1

    debug = True


class HelloWorldConfig(DefaultConfig):
    curriculum = [
        dict(order=[0, 1], T=2),
        dict(order=[0, 1, 0], T=3),
        dict(order=[0, 1, 0, 1], T=4)]
    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=32),
                                        MLP(),
                                        n_actions))
    log_name = 'hello_world'
    trainer = hello_world.HelloWorldTrainer()


class SimpleAdditionConfig(DefaultConfig):
    T = 30
    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
    ]

    n_controller_units = 64

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
    #     dict(width=9, n_digits=10, T=40),
    #     dict(width=10, n_digits=10, T=40),
    #     dict(width=11, n_digits=10, T=40),
    #     dict(width=12, n_digits=10, T=50),
    #     dict(width=13, n_digits=10, T=50),
    #     dict(width=15, n_digits=10, T=60),
    #     dict(width=17, n_digits=10, T=70),
    #     dict(width=20, n_digits=10, T=80),
    #     dict(width=25, n_digits=10, T=90),
    #     dict(width=30, n_digits=10, T=100),
    #     dict(width=35, n_digits=10, T=110),
    # ]

    log_name = 'simple_addition'

    trainer = simple_addition.SimpleAdditionTrainer()


class PointerConfig(DefaultConfig):
    T = 30
    curriculum = [
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10)]
    log_name = 'pointer'

    trainer = pointer_following.PointerTrainer()


class HardAdditionConfig(DefaultConfig):
    T = 40
    curriculum = [
        dict(height=2, width=3, n_digits=2, entropy_start=1.0),
        dict(height=2, width=3, n_digits=2, entropy_start=0.0)]
    log_name = 'hard_addition'
    trainer = hard_addition.HardAdditionTrainer()
    preserve_policy = False


class TranslatedMnistConfig(DefaultConfig):
    T = 10
    scaled = False
    discrete_attn = True
    curriculum = [
        dict(W=28, N=8, T=4),
        dict(W=28, N=8, T=10),
        dict(W=35, N=8, T=10),
        dict(W=45, N=8, T=10),
        dict(W=55, N=8, T=10),
        dict(W=65, N=8, T=12),
        dict(W=65, N=8, T=15),
        dict(W=65, N=8, T=20),
        dict(W=75, N=8, T=20),
        dict(W=85, N=8, T=20),
        dict(W=95, N=8, T=20)
    ]
    threshold = 0.10
    verbose = 4

    classifier_str = "MLP_50_50"

    @staticmethod
    def build_classifier(inp, outp_size):
        logits = MLP([50, 50], activation_fn=tf.nn.sigmoid)(inp, outp_size)
        return tf.nn.softmax(logits)

    # controller_func = staticmethod(
    #     lambda n_actions: FeedforwardCell(MLP([100, 100], activation_fn=tf.nn.sigmoid), n_actions))
    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=256),
                                        MLP(),
                                        n_actions))
    reward_window = 0.5

    log_name = 'translated_mnist'

    inc_delta = 0.1
    inc_x = 0.1
    inc_y = 0.1

    trainer = translated_mnist.TranslatedMnistTrainer()


class MnistArithmeticConfig(DefaultConfig):
    curriculum = [
        dict(W=100, N=16, T=20, n_digits=3, base=10),
    ]
    simple = False
    threshold = 0.15
    verbose = 4
    upper_bound = False

    classifier_str = "MLP_30_30"

    @staticmethod
    def build_classifier(inp, outp_size):
        logits = MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)
        return tf.nn.softmax(logits)

    # controller_func = staticmethod(
    #     lambda n_actions: FeedforwardCell(MLP([100, 100], activation_fn=tf.nn.sigmoid), n_actions))
    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=256),
                                        MLP(),
                                        n_actions))
    reward_window = 0.5

    log_name = 'mnist_arithmetic'

    inc_delta = 0.1
    inc_x = 0.1
    inc_y = 0.1

    trainer = mnist_arithmetic.MnistArithmeticTrainer()


class SimpleArithmeticConfig(DefaultConfig):
    curriculum = [dict(T=5)]
    mnist = False
    shape = (1, 2)
    op_loc = (0, 0)
    start_loc = (0, 0)
    n_digits = 1
    upper_bound = True
    base = 10
    batch_size = 32

    reward_window = 0.5

    n_controller_units = 128

    classifier_str = "MLP_30_30"

    @staticmethod
    def build_classifier(inp, outp_size):
        logits = MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)
        return tf.nn.softmax(logits)

    trainer = simple_arithmetic.SimpleArithmeticTrainer()

    controller_func = staticmethod(
        lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=64),
                                        MLP(),
                                        n_actions))
    log_name = 'simple_arithmetic'
    render_rollouts = staticmethod(simple_arithmetic.render_rollouts)


class TestConfig(DefaultConfig):
    n_val = 0
    n_test = 0
    batch_size = 1
    action_selection = IdentitySelect()
    verbose = 4

    action_seq = None

    @property
    def controller_func(self):
        return lambda n_actions: FixedController(self.action_seq, n_actions)

    @property
    def T(self):
        return len(self.action_seq)

    @property
    def n_train(self):
        return self.batch_size


class HelloWorldTest(TestConfig, HelloWorldConfig):
    order = [0, 1, 0]
    T = 3
    action_seq = [0, 1, 0]


class SimpleAdditionTest(TestConfig, SimpleAdditionConfig):
    width = 1
    n_digits = 10
    action_seq = [0, 2, 1, 1, 3, 5, 0]


class PointerTest(TestConfig, PointerConfig):
    width = 1
    n_digits = 10
    action_seq = [0, 2, 1]


class HardAdditionTest(TestConfig, HardAdditionConfig):
    width = 2
    height = 2
    n_digits = 10
    action_seq = [4, 2, 5, 6, 7, 0, 4, 3, 5, 6, 7, 0]


class TranslatedMnistTest(TestConfig, TranslatedMnistConfig):
    W = 60
    N = 14

    reward_window = 0.5

    inc_delta = 0.1
    inc_x = 0.1
    inc_y = 0.1

    discrete_attn = True

    classifier_str = "MLP_30_30"

    @staticmethod
    def build_classifier(inp, outp_size):
        logits = MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)
        return tf.nn.softmax(logits)

    action_seq = list(range(14))[::-1]
    batch_size = 16

    render_rollouts = staticmethod(translated_mnist.render_rollouts)


class MnistArithmeticTest(TestConfig, MnistArithmeticConfig):
    W = 100
    N = 8
    simple = False
    base = 3
    n_digits = 2

    reward_window = 0.5

    inc_delta = 0.1
    inc_x = 0.1
    inc_y = 0.1

    classifier_str = "MLP_30_30"

    @staticmethod
    def build_classifier(inp, outp_size):
        logits = MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)
        return tf.nn.softmax(logits)

    action_seq = range(14)
    batch_size = 16

    render_rollouts = staticmethod(translated_mnist.render_rollouts)


class SimpleArithmeticTest(TestConfig):
    mnist = False
    shape = (3, 3)
    op_loc = (0, 0)
    start_loc = (0, 0)
    n_digits = 3
    upper_bound = True
    base = 10
    batch_size = 16
    n_train = 16
    n_val = 0
    n_test = 0

    reward_window = 0.5

    classifier_str = "MLP_30_30"

    @staticmethod
    def build_classifier(inp, outp_size):
        logits = MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)
        return tf.nn.softmax(logits)

    action_seq = range(14)
    batch_size = 16

    trainer = simple_arithmetic.SimpleArithmeticTrainer()

    render_rollouts = staticmethod(simple_arithmetic.render_rollouts)


algorithms = dict(
    diff=DiffConfig(),
    reinforce=ReinforceConfig(),
    qlearning=QLearningConfig())


tasks = dict(
    hello_world=HelloWorldConfig(),
    simple_addition=SimpleAdditionConfig(),
    pointer=PointerConfig(),
    hard_addition=HardAdditionConfig(),
    translated_mnist=TranslatedMnistConfig(),
    mnist_arithmetic=MnistArithmeticConfig(),
    simple_arithmetic=SimpleArithmeticConfig())

test_configs = dict(
    hello_world=HelloWorldTest(),
    simple_addition=SimpleAdditionTest(),
    pointer=PointerTest(),
    hard_addition=HardAdditionTest(),
    translated_mnist=TranslatedMnistTest(),
    mnist_arithmetic=MnistArithmeticTest(),
    simple_arithmetic=SimpleArithmeticTest())
