import numpy as np
import tensorflow as tf
import os

from dps import cfg
from dps.utils import CompositeCell, MLP, FixedController, Config
from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.policy import (
    SoftmaxSelect, EpsilonGreedySelect, IdentitySelect)
from dps.mnist import LeNet, MNIST_CONFIG
from dps.experiments import (
    hello_world, simple_addition, pointer_following,
    hard_addition, translated_mnist, mnist_arithmetic, simple_arithmetic)


DEFAULT_CONFIG = Config(
    seed=12,

    preserve_policy=True,  # Whether to use the policy learned on the last stage of the curriculum for each new stage.

    optimizer_spec="rmsprop",
    build_updater=None,

    power_through=True,  # Whether to complete the entire curriculum, even if threshold not reached.
    slim=False,  # If true, tries to use little disk space
    max_steps=100,
    batch_size=100,
    n_train=10000,
    n_val=1000,
    n_test=0,

    threshold=1e-2,
    patience=np.inf,

    display_step=100,
    eval_step=10,
    checkpoint_step=1000,
    n_controller_units=32,

    controller=lambda n_actions: CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=32), MLP(), n_actions),

    lr_schedule="exponential 0.001 1000 0.96",
    exploration_schedule="exponential 10.0 1000 0.96",
    noise_schedule=None,

    test_time_explore=None,

    max_grad_norm=0.0,
    l2_norm_penalty=0.0,
    gamma=1.0,
    reward_window=0.1,

    n_auxiliary_tasks=0,
    auxiliary_coef=0,

    debug=False,
    verbose=False,
    display=False,
    save_display=False,
    path=os.getcwd(),
    max_time=0,
)


cfg._stack.append(DEFAULT_CONFIG)


REINFORCE_CONFIG = Config(
    build_updater=REINFORCE,
    action_selection=lambda na: SoftmaxSelect(na),
    test_time_explore=0.1,
    patience=np.inf,
    entropy_schedule="exp 0.01 100000 0.1",
    lr_schedule="constant 0.001",
    exploration_schedule="exp 10.0 100000 1.0",
    gamma=0.99,
)


QLEARNING_CONFIG = Config(
    build_updater=QLearning,
    action_selection=lambda n_actions: EpsilonGreedySelect(n_actions),
    double=False,

    lr_schedule="exponential 0.00025 1000 1.0",
    exploration_schedule="polynomial 1.0 10000 0.1 1",

    optimizer_spec="rmsprop",

    replay_max_size=1000000,
    replay_threshold=-0.5,
    replay_proportion=0.0,
    gamma=0.99,

    target_update_rate=None,
    steps_per_target_update=10000,
    recurrent=True,
    patience=np.inf,
    samples_per_update=32,  # Number of rollouts between parameter updates
    update_batch_size=32,  # Number of sample rollouts to use for each parameter update
    batch_size=64,  # Number of sample experiences to execute
    test_time_explore=0.05,

    l2_norm_penalty=0.0,
    max_grad_norm=0.0,

    n_controller_units=256,
)


DQN_CONFIG = QLEARNING_CONFIG.copy(
    # From Nature paper

    # Rewards are clipped: all negative rewards set to -1, all positive set to 1, 0 unchanged.
    batch_size=32,

    lr_schedule="0.00025",

    # annealed linearly from 1 to 0.1 over first million frames,
    # fixed at 0.1 thereafter "
    exploration_schedule="polynomial 1.0 10000 0.1 1",

    replay_max_size=1e6,

    # max number of frames/states: 10 million

    test_time_explore=0.05,
    # target_network_update Once every 10000 frames
    steps_per_target_update=10000,

    gamma=0.99,

    # 4 actions selected between each update
    # RMS prop momentum: 0.95
    # squared RMS prop gradient momentum: 0.95
    # min squared gradient (RMSProp): 0.01
    # exploration 1 to 0.1 over 1,000,000 frames
    # total number of frames: 50,000,000, but because of frame skip, equivalent to 200,000,000 frames
)


FPS_CONFIG = QLEARNING_CONFIG.copy(
    gamma=0.99,
    update_batch_size=32,
    batch_size=64,

    # Exploration: annealed linearly from 1 to 0.1 over first million steps, fixed at 0.1 thereafter

    # Replay max size: 1million *frames*

    # They actually update every 4 *steps*, rather than every 4 experiences
    samples_per_update=4,
)


DUELING_CONFIG = QLEARNING_CONFIG.copy(
    max_grad_norm=10.0,
    lr_schedule="6.25e-5",  # when prioritized experience replay was used
    test_time_explore=0.001,  # fixed at this value...this might also be the training exploration, its not clear
)


DOUBLE_CONFIG = QLEARNING_CONFIG.copy(
    double=True,
    exploration_start=0.1,  # Not totally clear, but seems like they use the same scheme as DQN, but go from 1 to 0.01, instead of 1 to 0.1
    test_time_explore=0.001,
    # Target network update rate: once every 30,000 frames (DQN apparently does it once every 10,000 frames).
)


DEBUG_CONFIG = Config(
    max_steps=4,
    batch_size=2,
    n_train=10,
    n_val=10,
    n_test=0,

    display_step=1,
    eval_step=1,

    debug=True,
)


HELLO_WORLD_CONFIG = Config(
    build_env=hello_world.build_env,
    curriculum=[
        dict(order=[0, 1], T=2),
        dict(order=[0, 1, 0], T=3),
        dict(order=[0, 1, 0, 1], T=4)],
    controller=lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=32), MLP(), n_actions),
    log_name='hello_world',
)


SIMPLE_ADDITION_CONFIG = Config(
    build_env=simple_addition.build_env,
    T=30,
    curriculum=[
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10),
        dict(width=3, n_digits=10),
    ],

    n_controller_units=64,

    # curriculum=[
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

    log_name='simple_addition',
)


POINTER_CONFIG = Config(
    build_env=pointer_following.build_env,
    T=30,
    curriculum=[
        dict(width=1, n_digits=10),
        dict(width=2, n_digits=10)],
    log_name='pointer',
)


HARD_ADDITION_CONFIG = Config(
    build_env=hard_addition.build_env,
    T=40,
    curriculum=[
        dict(height=2, width=3, n_digits=2, entropy_start=1.0),
        dict(height=2, width=3, n_digits=2, entropy_start=0.0)],
    log_name='hard_addition',
)


TRANSLATED_MNIST_CONFIG = Config(
    build_env=translated_mnist.build_env,

    T=10,
    scaled=False,
    discrete_attn=True,
    curriculum=[
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
    ],
    threshold=0.10,
    verbose=4,

    classifier_str="MLP_50_50",
    build_classifier=lambda inp, outp_size, is_training=False: tf.nn.softmax(
        MLP([50, 50], activation_fn=tf.nn.sigmoid)(inp, outp_size)),

    controller=lambda n_actions: CompositeCell(tf.contrib.rnn.LSTMCell(num_units=256), MLP(), n_actions),
    reward_window=0.5,

    log_name='translated_mnist',

    inc_delta=0.1,
    inc_x=0.1,
    inc_y=0.1,
)


MNIST_ARITHMETIC_CONFIG = Config(
    build_env=mnist_arithmetic.build_env,

    curriculum=[
        dict(W=100, N=16, T=20, n_digits=3, base=10),
    ],
    simple=False,
    threshold=0.15,
    verbose=4,
    upper_bound=False,

    classifier_str="MLP_30_30",
    build_classifier=lambda inp, outp_size, is_training=False: tf.nn.softmax(
        MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)),

    controller=lambda n_actions: CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=256), MLP(), n_actions),
    reward_window=0.5,

    log_name='mnist_arithmetic',

    inc_delta=0.1,
    inc_x=0.1,
    inc_y=0.1,
)


SIMPLE_ARITHMETIC_CONFIG = Config(
    build_env=simple_arithmetic.build_env,

    curriculum=[
        # dict(T=30),
        dict(T=10, n_digits=2, shape=(1, 3)),
        # dict(T=3),
        # dict(T=4),
        # dict(T=5),
        # dict(T=5, shape=(1, 3), display=True),
        # dict(T=5, n_digits=2, shape=(1, 3), display=True),
        # dict(T=10, n_digits=2, shape=(1, 3), display=True),
        # dict(T=10, n_digits=2, shape=(2, 2), display=True),
        # dict(T=15, shape=(3, 3), n_digits=2, display=True),
        # dict(T=5, shape=(1, 3)),
        # dict(T=7, shape=(1, 3), display=False),
        # dict(T=7, shape=(1, 3), n_digits=2, display=False),
        # dict(T=7, shape=(1, 4), n_digits=2, display=False),
        # dict(T=8, shape=(1, 5), n_digits=2, display=False),
        # dict(T=9, shape=(1, 6), n_digits=2, display=False),
        # dict(T=10, shape=(1, 7), n_digits=2, display=False),
        # dict(T=11, shape=(1, 7), n_digits=2, display=False),
        # dict(T=12, shape=(1, 7), n_digits=2, display=False),
        # dict(T=13, shape=(1, 7), n_digits=2, display=False),
        # dict(T=14, shape=(1, 7), n_digits=2, display=False),
        # dict(T=15, shape=(1, 7), n_digits=2, display=False),
        # dict(T=15, shape=(1, 7), n_digits=3, display=False),
        # dict(T=15, shape=(1, 7), n_digits=4, display=False),
        # dict(T=15, shape=(1, 7), n_digits=5, display=False),
        # dict(T=15, shape=(1, 7), n_digits=6, display=True),
    ],
    display=False,
    mnist=True,
    shape=(1, 2),
    op_loc=(0, 0),
    start_loc=(0, 0),
    n_digits=1,
    upper_bound=False,
    base=10,
    batch_size=64,
    threshold=0.04,

    reward_window=0.5,

    n_controller_units=64,

    classifier_str="LeNet2_1024",
    build_classifier=lambda inp, output_size, is_training=False: tf.nn.softmax(
        LeNet(1024, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)),

    mnist_config=MNIST_CONFIG.copy(
        eval_step=100,
        max_steps=100000,
        patience=np.inf,
        threshold=0.05,
        include_blank=True),

    log_name='simple_arithmetic',
    render_rollouts=simple_arithmetic.render_rollouts,
)


def adjust_for_test(config):
    try:
        n_train = config.batch_size
    except AttributeError:
        n_train = 1
    config.update(
        T=len(config.action_seq),
        controller=lambda n_actions: FixedController(config.action_seq, n_actions),
        batch_size=n_train,
        n_train=n_train,
        n_val=0,
        n_test=0,
        action_selection=lambda na: IdentitySelect(na),
        verbose=4,
    )


HELLO_WORLD_TEST = HELLO_WORLD_CONFIG.copy(
    batch_size=10,
    order=[0, 1, 0],
    T=3,
    action_seq=[0, 1, 0],
)
adjust_for_test(HELLO_WORLD_TEST)


SIMPLE_ADDITION_TEST = SIMPLE_ADDITION_CONFIG.copy(
    width=1,
    n_digits=10,
    action_seq=[0, 2, 1, 1, 3, 5, 0],
)
adjust_for_test(SIMPLE_ADDITION_TEST)


POINTER_TEST = POINTER_CONFIG.copy(
    width=1,
    n_digits=10,
    action_seq=[0, 2, 1],
)
adjust_for_test(POINTER_TEST)


HARD_ADDITION_TEST = HARD_ADDITION_CONFIG.copy(
    width=2,
    height=2,
    n_digits=10,
    action_seq=[4, 2, 5, 6, 7, 0, 4, 3, 5, 6, 7, 0],
)
adjust_for_test(HARD_ADDITION_TEST)


TRANSLATED_MNIST_TEST = TRANSLATED_MNIST_CONFIG.copy(
    W=60,
    N=14,

    reward_window=0.5,

    inc_delta=0.1,
    inc_x=0.1,
    inc_y=0.1,

    discrete_attn=True,

    classifier_str="MLP_30_30",
    build_classifier=lambda inp, outp_size, is_training=False: tf.nn.softmax(
        MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)),

    action_seq=list(range(14))[::-1],
    batch_size=16,

    render_rollouts=translated_mnist.render_rollouts,
)
adjust_for_test(TRANSLATED_MNIST_TEST)


MNIST_ARITHMETIC_TEST = MNIST_ARITHMETIC_CONFIG.copy(
    W=100,
    N=8,
    simple=False,
    base=3,
    n_digits=2,

    reward_window=0.5,

    inc_delta=0.1,
    inc_x=0.1,
    inc_y=0.1,

    classifier_str="MLP_30_30",
    build_classifier=lambda inp, outp_size, is_training=False: tf.nn.softmax(
        MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)),

    action_seq=range(14),
    batch_size=16,

    render_rollouts=translated_mnist.render_rollouts,
)
adjust_for_test(MNIST_ARITHMETIC_TEST)


SIMPLE_ARITHMETIC_TEST = SIMPLE_ARITHMETIC_CONFIG.copy(
    mnist=True,
    shape=(5, 5),
    op_loc=(0, 0),
    start_loc=(0, 0),
    n_digits=3,
    upper_bound=True,
    base=10,
    n_examples=1,
    batch_size=1,
    n_train=1,
    n_val=0,
    n_test=0,
    curriculum=[dict(T=10, n_digits=2, shape=(1, 3))],

    reward_window=0.5,

    mnist_config=MNIST_CONFIG.copy(
        eval_step=100, max_steps=100000, patience=np.inf, threshold=0.01, include_blank=True),

    classifier_str="LeNet2_1024",
    build_classifier=lambda inp, output_size, is_training=False: tf.nn.softmax(
        LeNet(1024, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)),

    action_seq=range(simple_arithmetic.SimpleArithmetic.n_actions),
    render_rollouts=simple_arithmetic.render_rollouts,
)
adjust_for_test(SIMPLE_ARITHMETIC_TEST)


algorithms = dict(
    reinforce=REINFORCE_CONFIG,
    qlearning=QLEARNING_CONFIG)


tasks = dict(
    hello_world=HELLO_WORLD_CONFIG,
    simple_addition=SIMPLE_ADDITION_CONFIG,
    pointer=POINTER_CONFIG,
    hard_addition=HARD_ADDITION_CONFIG,
    translated_mnist=TRANSLATED_MNIST_CONFIG,
    mnist_arithmetic=MNIST_ARITHMETIC_CONFIG,
    simple_arithmetic=SIMPLE_ARITHMETIC_CONFIG)

test_configs = dict(
    hello_world=HELLO_WORLD_TEST,
    simple_addition=SIMPLE_ADDITION_TEST,
    pointer=POINTER_TEST,
    hard_addition=HARD_ADDITION_TEST,
    translated_mnist=TRANSLATED_MNIST_TEST,
    mnist_arithmetic=MNIST_ARITHMETIC_TEST,
    simple_arithmetic=SIMPLE_ARITHMETIC_TEST)
