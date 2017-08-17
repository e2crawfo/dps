import numpy as np
import tensorflow as tf

from dps import cfg
from dps.utils import (
    CompositeCell, FeedforwardCell, MLP, FixedDiscreteController,
    FixedController, DpsConfig, Config)
from dps.rl import (
    RLUpdater, rl_render_hook, REINFORCE, PPO, TRPO, RobustREINFORCE, QLearning, TrustRegionQLearning, ProximalQLearning)
from dps.rl.qlearning import DuelingHead
from dps.rl.policy import (
    Policy, Softmax, EpsilonSoftmax, EpsilonGreedy, Deterministic,
    Normal, Gamma, ProductDist)
from dps.vision import LeNet, MNIST_CONFIG
from dps.experiments import (
    hello_world, room, grid, grid_bandit, path_discovery, simple_addition, pointer_following,
    hard_addition, translated_mnist, mnist_arithmetic, simple_arithmetic,
    alt_arithmetic)
from dps.rl.value import actor_critic, TrustRegionPolicyEvaluation, ProximalPolicyEvaluation, PolicyEvaluation


def softmax(env):
    return Softmax(env.n_actions)


def epsilon_greedy(env):
    return EpsilonGreedy(env.n_actions)


class LstmController(object):
    def __call__(self, n_params, name=None):
        return CompositeCell(
            tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
            MLP(), n_params, name=name)


class DuelingLstmController(object):
    def __call__(self, n_params, name=None):
        return CompositeCell(
            tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
            DuelingHead(MLP(), MLP()),
            n_params,
            name=name)


class FeedforwardController(object):
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, n_params, name=None):
        return FeedforwardCell(MLP(*self.args, **self.kwargs), n_paramsname=name)


def get_updater(env):
    with cfg.actor_config:
        action_selection = cfg.action_selection(env)
        policy_controller = cfg.controller(
            action_selection.n_params, name="actor_controller")

    if 'critic_config' in cfg and cfg.critic_config.alg is not None:
        with cfg.critic_config:
            critic_controller = cfg.controller(1, name="critic_controller")

        return actor_critic(
            env, policy_controller, action_selection, critic_controller,
            cfg.actor_config, cfg.critic_config)
    else:
        with cfg.actor_config:
            policy = Policy(policy_controller, action_selection, env.obs_shape)
            updater = RLUpdater(env, policy, cfg.alg(policy, name="actor"))
            return updater


def get_experiment_name():
    name = []

    try:
        name.append('actor={}'.format(cfg.actor_config.name))
    except:
        pass

    try:
        name.append('critic={}'.format(cfg.critic_config.name))
    except:
        pass

    if not name:
        try:
            name.append('name={}'.format(cfg.name))
        except:
            pass

    try:
        name.append('seed={}'.format(cfg.seed))
    except:
        pass
    return '_'.join(name)


DEFAULT_CONFIG = DpsConfig(
    name="Default",
    seed=None,

    preserve_policy=True,  # Whether to use the policy learned on the last stage of the curriculum for each new stage.
    power_through=True,  # Whether to complete the entire curriculum, even if threshold not reached.

    optimizer_spec="adam",
    get_updater=get_updater,

    slim=False,  # If true, tries to use little disk space
    max_steps=10000,
    batch_size=32,
    n_train=10000,
    n_val=1000,
    threshold=1e-2,
    patience=np.inf,

    render_step=np.inf,
    display_step=1000,
    eval_step=10,

    noise_schedule=None,
    test_time_explore=-1.,
    max_grad_norm=0.0,
    reward_window=0.1,
    exploration_schedule=10.,
    lr_schedule='0.001',

    lmbda=1.0,
    gamma=1.0,

    standardize_advantage=True,
    reset_env=True,

    dense_reward=False,

    max_time=0,

    n_controller_units=32,
    controller=LstmController(),
    action_selection=softmax,

    deadline='',
    render_hook=rl_render_hook,

    get_experiment_name=get_experiment_name,
)


cfg._stack.append(DEFAULT_CONFIG)

# Critic configs.
BASELINE_CONFIG = Config(
    name="Baseline",
    alg=None
)

PE_CONFIG = Config(
    name="PolicyEvaluation",
    alg=PolicyEvaluation,
    lr_schedule=1e-2,
    optimizer_spec='adam',
    opt_steps_per_batch=10,
)


PPE_CONFIG = Config(
    name="ProximalPolicyEvaluation",
    alg=ProximalPolicyEvaluation,
    lr_schedule=1e-2,
    optimizer_spec='adam',
    epsilon=0.2,
    opt_steps_per_batch=10,
    S=5,
)


TRPE_CONFIG = Config(
    name="TRPE",
    alg=TrustRegionPolicyEvaluation,
    delta_schedule='0.01',
    max_cg_steps=10,
    max_line_search_steps=10,
)


# Actor configs.
REINFORCE_CONFIG = Config(
    name="REINFORCE",
    alg=REINFORCE,
    batch_size=16,
    entropy_schedule="0.1",
    lr_schedule="1e-2",
    # lr_schedule="1e-4",
    # lr_schedule="poly 1e-4 100000 1e-6 1",  # also good
    n_controller_units=64,
    # exploration_schedule='exp 10.0 100000 0.1',
    # exploration_schedule='poly 10.0 10000 1e-6 1.0',
    test_time_explore=0.1
)


PPO_CONFIG = Config(
    name="PPO",
    alg=PPO,
    entropy_schedule="0.1",
    epsilon=0.2,
    opt_steps_per_batch=10,
    lr_schedule="1e-3",
    n_controller_units=64,
    # exploration_schedule='poly 1.0 10000 1e-6 1.0',
    test_time_explore=-1
)


TRPO_CONFIG = Config(
    name="TRPO",
    alg=TRPO,
    entropy_schedule='0.1',
    # exploration_schedule="poly 1.0 10000 1e-6 1.0",
    max_cg_steps=10,
    max_line_search_steps=20,
    delta_schedule="0.01"
)


ROBUST_CONFIG = Config(
    name="RobustREINFORCE",
    alg=RobustREINFORCE,
    entropy_schedule="0.1",
    # exploration_schedule="10.0",
    max_line_search_steps=10,
    delta_schedule="0.01",
    max_cg_steps=0,
)


QLEARNING_CONFIG = Config(
    name="QLearning",
    alg=QLearning,

    action_selection=epsilon_greedy,
    n_controller_units=64,
    controller=DuelingLstmController(),
    double=True,

    lr_schedule="0.00025",
    exploration_schedule="pwc 1.0 40000 0.1",
    test_time_explore="0.01",

    optimizer_spec="adam",

    gamma=1.0,

    init_steps=5000,

    opt_steps_per_batch=10,
    target_update_rate=0.01,
    steps_per_target_update=None,
    patience=np.inf,
    update_batch_size=32,  # Number of sample rollouts to use for each parameter update
    batch_size=1,  # Number of sample experiences per update

    replay_max_size=20000,
    alpha=0.7,
    beta_schedule="0.5",

    max_grad_norm=0.0,
)


TRQL_CONFIG = QLEARNING_CONFIG.copy(
    name="TrustRegionQLearning",
    alg=TrustRegionQLearning,
    delta_schedule=0.01,
    max_cg_steps=10,
    max_line_search_steps=20,
)


PQL_CONFIG = QLEARNING_CONFIG.copy(
    name="ProximalQLearning",
    alg=ProximalQLearning,
    opt_steps_per_batch=10,
    S=1,
    epsilon=0.2,
)


DQN_CONFIG = QLEARNING_CONFIG.copy(
    # From Nature paper

    # Rewards are clipped: all negative rewards set to -1, all positive set to 1, 0 unchanged.
    batch_size=32,

    lr_schedule="0.01",
    # lr_schedule="0.00025",

    # annealed linearly from 1 to 0.1 over first million frames,
    # fixed at 0.1 thereafter "
    exploration_schedule="polynomial 1.0 10000 0.1 1",

    replay_max_size=1e6,

    # max number of frames/states: 10 million

    test_time_explore=0.05,
    # Once every 10000 frames
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

    display_step=1,
    eval_step=1,

    debug=True,
)


order = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
HELLO_WORLD_CONFIG = Config(
    build_env=hello_world.build_env,
    curriculum=[
        dict(order=order[:2], T=2),
        dict(order=order[:3], T=3),
        dict(order=order[:4], T=4),
        dict(order=order[:5], T=5),
        dict(order=order[:6], T=6),
        dict(order=order[:7], T=7),
        dict(order=order[:8], T=8),
        dict(order=order[:9], T=9),
    ],
    log_name='hello_world',
)


def room_action_selection(env):
    if cfg.room_angular:
        return ProductDist(Normal(), Normal(), Gamma())
    else:
        return ProductDist(Normal(), Normal())


ROOM_CONFIG = Config(
    build_env=room.build_env,
    curriculum=[dict(T=20)],
    # curriculum=[dict(T=20), dict(T=10), dict(T=5)],
    n_controller_units=32,
    action_selection=room_action_selection,
    log_name='room',
    eval_step=10,
    T=20,
    batch_size=10,
    dense_reward=False,
    restart_prob=0.0,
    max_step=0.1,
    room_angular=False,
    l2l=False,
    reward_radius=0.2,
)


GRID_CONFIG = Config(
    build_env=grid.build_env,
    curriculum=[dict()],
    n_controller_units=32,
    log_name='grid',
    eval_step=10,
    batch_size=10,
    dense_reward=True,
    restart_prob=0.0,
    l2l=False,
    shape=(5, 5),
    T=20
)


GRID_BANDIT_CONFIG = Config(
    build_env=grid_bandit.build_env,
    curriculum=[dict()],
    n_controller_units=32,
    log_name='grid_bandit',
    eval_step=10,
    batch_size=32,
    shape=(2, 2),
    T=5,
    n_arms=10,
    threshold=-5,
)


PATH_DISCOVERY_CONFIG = Config(
    build_env=path_discovery.build_env,
    curriculum=[
        dict(shape=(2, 2), threshold=-6),
        dict(shape=(3, 3), threshold=-4),
        dict(shape=(4, 4), threshold=-2)
    ],
    n_controller_units=32,
    log_name='path_discovery',
    eval_step=10,
    batch_size=32,
    shape=(3, 3),
    T=10,
    threshold=-10,
)


SIMPLE_ADDITION_CONFIG = Config(
    dense_reward=False,
    build_env=simple_addition.build_env,
    T=30,
    curriculum=[
        dict(width=1, base=10, threshold=0.01),
        dict(width=2, base=10, threshold=0.01),
        dict(width=3, base=10, threshold=0.01),
    ],
    # curriculum=[
    #     dict(width=1, base=10, threshold=6.0),
    #     dict(width=2, base=10, threshold=8.0),
    #     dict(width=3, base=10, threshold=10.0),
    # ],

    n_controller_units=32,
    log_name='simple_addition',
)


POINTER_CONFIG = Config(
    build_env=pointer_following.build_env,
    T=30,
    dense_reward=True,
    curriculum=[
        dict(width=1, base=10),
        dict(width=2, base=10)],
    log_name='pointer',
)


HARD_ADDITION_CONFIG = Config(
    build_env=hard_addition.build_env,
    T=40,
    curriculum=[
        dict(height=2, width=3, base=2, entropy_start=1.0),
        dict(height=2, width=3, base=2, entropy_start=0.0)],
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

    n_controller_units=64,
    reward_window=0.5,

    log_name='translated_mnist',

    inc_delta=0.1,
    inc_x=0.1,
    inc_y=0.1,
)


MNIST_ARITHMETIC_CONFIG = Config(
    build_env=mnist_arithmetic.build_env,

    curriculum=[
        dict(W=100, N=16, T=20, min_digits=2, max_digits=3, base=10),
    ],
    simple=False,
    threshold=0.15,
    verbose=4,

    classifier_str="MLP_30_30",
    build_classifier=lambda inp, outp_size, is_training=False: tf.nn.softmax(
        MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)),

    n_controller_units=256,
    reward_window=0.5,

    log_name='mnist_arithmetic',

    inc_delta=0.1,
    inc_x=0.1,
    inc_y=0.1,
)


SIMPLE_ARITHMETIC_CONFIG = Config(
    build_env=simple_arithmetic.build_env,
    symbols=[
        ('A', lambda x: sum(x)),
        ('M', lambda x: np.product(x)),
        ('C', lambda x: len(x))],

    curriculum=[
        dict(T=30, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    display=False,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    batch_size=64,
    threshold=0.04,

    reward_window=0.4,

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


def alt_arithmetic_action_selection(env):
    if cfg.ablation == 'bad_wiring':
        return ProductDist(Softmax(11), Normal(), Normal(), Normal())
    elif cfg.ablation == 'no_classifiers':
        return ProductDist(Softmax(9), Softmax(10, one_hot=0), Softmax(10, one_hot=0), Softmax(10, one_hot=0))
    elif cfg.ablation == 'no_ops':
        return ProductDist(Softmax(11), Normal(), Normal(), Normal())
    elif cfg.ablation == 'no_modules':
        return ProductDist(Softmax(11), Normal(), Normal(), Normal())
    else:
        return Softmax(env.n_actions)


ALT_ARITHMETIC_CONFIG = Config(
    build_env=alt_arithmetic.build_env,
    symbols=[('A', lambda x: sum(x)), ('M', lambda x: np.product(x)), ('C', lambda x: len(x))],

    curriculum=[
        dict(T=20, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    force_2d=False,
    display=False,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    batch_size=64,
    threshold=0.04,

    dense_reward=True,
    reward_window=0.4,

    n_controller_units=64,

    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.

    classifier_str="LeNet2_1024",
    build_classifier=lambda inp, output_size, is_training=False: tf.nn.softmax(
        LeNet(1024, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)),

    mnist_config=MNIST_CONFIG.copy(
        eval_step=100,
        max_steps=100000,
        patience=np.inf,
        threshold=0.05,
        include_blank=True),

    log_name='alt_arithmetic',
    render_rollouts=None
)


def adjust_for_test(config):
    try:
        n_train = config.batch_size
    except AttributeError:
        n_train = 1
    config.update(
        T=len(config.action_seq),
        controller=lambda n_params, name=None: FixedDiscreteController(config.action_seq, n_params, name=name),
        batch_size=n_train,
        n_train=n_train,
        n_val=0,
        action_selection=lambda env: Deterministic(env.n_actions),
        verbose=4,
    )


HELLO_WORLD_TEST = HELLO_WORLD_CONFIG.copy(
    batch_size=10,
    order=[0, 1, 0],
    T=3,
    action_seq=[0, 1, 0],
)
adjust_for_test(HELLO_WORLD_TEST)


ROOM_CONFIG_TEST = ROOM_CONFIG.copy(
    build_env=room.build_env,
    T=6,
    controller=lambda n_params, name=None: FixedController(
        np.concatenate(
            [0.1 * np.ones((6, 1)), 0.1 * np.ones((6, 1))], axis=1),
        name=name
    ),
    action_selection=lambda env: Deterministic(env.n_actions),
    batch_size=2,
    room_angular=False,
    n_train=2,
    n_val=0,
    verbose=4,
    dense_reward=True,
)


GRID_BANDIT_CONFIG_TEST = Config(
    shape=(2, 2),
    T=10,
    action_seq=[4, 1, 4, 2, 4, 3, 4, 5, 6],
    build_env=grid_bandit.build_env,
    n_arms=3
)
adjust_for_test(GRID_BANDIT_CONFIG_TEST)


SIMPLE_ADDITION_TEST = SIMPLE_ADDITION_CONFIG.copy(
    width=1,
    base=10,
    action_seq=[0, 2, 1, 1, 3, 5, 0],
)
adjust_for_test(SIMPLE_ADDITION_TEST)


POINTER_TEST = POINTER_CONFIG.copy(
    width=1,
    base=10,
    action_seq=[0, 2, 1],
)
adjust_for_test(POINTER_TEST)


HARD_ADDITION_TEST = HARD_ADDITION_CONFIG.copy(
    width=2,
    height=2,
    base=10,
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
    min_digits=2,
    max_digits=3,

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
    base=10,
    n_examples=1,
    batch_size=1,
    n_train=1,
    n_val=0,
    curriculum=[dict(T=10, min_digits=2, max_digits=3, shape=(1, 3))],

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


ALT_ARITHMETIC_TEST = ALT_ARITHMETIC_CONFIG.copy(
    shape=(3, 1),
    op_loc=(0, 0),
    start_loc=(0, 0),
    min_digits=2,
    max_digits=3,
    base=10,
    batch_size=16,
    threshold=0.04,

    reward_window=0.4,

    n_controller_units=64,

    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.

    classifier_str="LeNet2_1024",
    build_classifier=lambda inp, output_size, is_training=False: tf.nn.softmax(
        LeNet(1024, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)),

    mnist_config=MNIST_CONFIG.copy(
        eval_step=100,
        max_steps=100000,
        patience=np.inf,
        threshold=0.05,
        include_blank=True),

    log_name='alt_arithmetic',
    render_rollouts=None,
    action_seq=[5, 2, 4, 9, 2, 4, 8]
)
adjust_for_test(ALT_ARITHMETIC_TEST)


critic_configs = dict(
    trpe=TRPE_CONFIG,
    ppe=PPE_CONFIG,
    pe=PE_CONFIG,
    baseline=BASELINE_CONFIG
)


actor_configs = dict(
    reinforce=REINFORCE_CONFIG,
    trpo=TRPO_CONFIG,
    ppo=PPO_CONFIG,
    robust=ROBUST_CONFIG,
    qlearning=QLEARNING_CONFIG,
    trql=TRQL_CONFIG,
    pql=PQL_CONFIG
)


tasks = dict(
    hello_world=HELLO_WORLD_CONFIG,
    room=ROOM_CONFIG,
    grid=GRID_CONFIG,
    grid_bandit=GRID_BANDIT_CONFIG,
    path_discovery=PATH_DISCOVERY_CONFIG,
    simple_addition=SIMPLE_ADDITION_CONFIG,
    pointer=POINTER_CONFIG,
    hard_addition=HARD_ADDITION_CONFIG,
    translated_mnist=TRANSLATED_MNIST_CONFIG,
    mnist_arithmetic=MNIST_ARITHMETIC_CONFIG,
    simple_arithmetic=SIMPLE_ARITHMETIC_CONFIG,
    alt_arithmetic=ALT_ARITHMETIC_CONFIG
)


test_configs = dict(
    hello_world=HELLO_WORLD_TEST,
    room=ROOM_CONFIG_TEST,
    grid_bandit=GRID_BANDIT_CONFIG_TEST,
    simple_addition=SIMPLE_ADDITION_TEST,
    pointer=POINTER_TEST,
    hard_addition=HARD_ADDITION_TEST,
    translated_mnist=TRANSLATED_MNIST_TEST,
    mnist_arithmetic=MNIST_ARITHMETIC_TEST,
    simple_arithmetic=SIMPLE_ARITHMETIC_TEST,
    alt_arithmetic=ALT_ARITHMETIC_TEST
)


def parse_task_actor_critic(task, actor, critic):
    task = [t for t in tasks if t.startswith(task)]
    assert len(task) == 1, "Ambiguity in task selection, possibilities are: {}.".format(task)
    task = task[0]

    _actors = list(actor_configs) + ['visualize']
    actor = [a for a in _actors if a.startswith(actor)]
    assert len(actor) == 1, "Ambiguity in actor selection, possibilities are: {}.".format(actor)
    actor = actor[0]

    critic = [c for c in list(critic_configs) if c.startswith(critic)]
    assert len(critic) == 1, "Ambiguity in critic selection, possibilities are: {}.".format(critic)
    critic = critic[0]

    return task, actor, critic
