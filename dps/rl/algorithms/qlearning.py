import tensorflow as tf

from dps import cfg
from dps.utils import Config, MLP, CompositeCell
from dps.rl import (
    RLContext, RLObject, Agent, StochasticGradientDescent, BuildEpsilonGreedyPolicy,
    DiscretePolicy, RLUpdater, ActionValueFunction,
    PolicyEvaluation_StateAction, Retrace, PrioritizedReplayBuffer
)


class DuelingHead(object):
    def __init__(self, value_fn, adv_fn, kind="mean"):
        self.value_fn, self.adv_fn = value_fn, adv_fn

        if kind not in "mean max".split():
            raise Exception('`kind` must be one of "mean", "max".')
        self.kind = kind

    def __call__(self, inp, n_actions):
        value = self.value_fn(inp, 1)
        advantage = self.adv_fn(inp, n_actions)

        if self.kind == "mean":
            normed_advantage = advantage - tf.reduce_mean(advantage, axis=-1, keep_dims=True)
        else:
            normed_advantage = advantage - tf.reduce_max(advantage, axis=-1, keep_dims=True)
        return value + normed_advantage


class BuildDuelingLstmController(object):
    def __call__(self, params_dim, name=None):
        return CompositeCell(
            tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
            DuelingHead(MLP(), MLP()),
            params_dim,
            name=name)


class AgentUpdater(RLObject):
    def __init__(self, parameter, main_agent, target_agent, name="update_target_agent"):
        self.parameter = parameter
        self.main_agent = main_agent
        self.target_agent = target_agent
        self.name = name
        self.steps_since_target_update = 0

    def _make_update_op(self, v_source, v_target):
        return v_target.assign(v_source)

    def build_core_signals(self, context):
        main_agent_variables = self.main_agent.trainable_variables()
        target_agent_variables = self.target_agent.trainable_variables()

        with tf.name_scope(self.name):
            target_agent_update = []

            for v_source, v_target in zip(main_agent_variables, target_agent_variables):
                update_op = self._make_update_op(v_source, v_target)
                target_agent_update.append(update_op)

            self.target_agent_update = tf.group(*target_agent_update)

    def post_update(self, feed_dict, context):
        if self.steps_since_target_update > self.parameter:
            tf.get_default_session().run(self.target_agent_update)
            self.steps_since_target_update = 0
        else:
            self.steps_since_target_update += 1


class SmoothAgentUpdater(AgentUpdater):
    def _make_update_op(self, v_source, v_target):
        return v_target.assign_sub(self.parameter * (v_target - v_source))

    def post_update(self, feed_dict, context):
        tf.get_default_session().run(self.target_agent_update)


class PriorityFunc(object):
    def __init__(self, value_function):
        self.value_function = value_function

    def __call__(self, context):
        priority_signal = context.get_signal('one_step_td_errors', self.value_function)
        priority_signal = tf.reduce_sum(tf.abs(priority_signal), axis=[0, -1])
        return priority_signal


class MaxPriorityFunc(PriorityFunc):
    def __call__(self, context):
        priority_signal = context.get_signal('one_step_td_errors', self.value_function)
        priority_signal = tf.reduce_max(tf.abs(priority_signal), axis=[0, -1])
        return priority_signal


def QLearning(env):
    with RLContext(cfg.gamma) as context:
        policy = cfg.build_policy(env, name="actor")
        assert isinstance(policy, DiscretePolicy), "QLearning currently only works with discrete policies."

        context.set_behaviour_policy(policy)

        agent = Agent("agent", cfg.build_controller, [policy])

        start, end = agent._head_offsets["actor"]
        action_value_function = ActionValueFunction(env.n_actions, policy, "q")
        agent.add_head(action_value_function, start, end)

        target_agent = agent.deepcopy("target_agent")
        target_agent['actor'].exploration_schedule = cfg.target_exploration_schedule

        assert not (cfg.double and cfg.reverse_double)

        if cfg.double:
            learn_about_policy = policy
            backup_values = target_agent['q']
        elif cfg.reverse_double:
            learn_about_policy = target_agent['actor']
            backup_values = agent['q']
        else:
            learn_about_policy = target_agent['actor']
            backup_values = target_agent['q']

        action_values_from_returns = Retrace(
            learn_about_policy, backup_values, lmbda=cfg.lmbda,
            to_action_value=True, from_action_value=True,
            name="RetraceQ"
        )

        priority_func = MaxPriorityFunc(action_value_function)
        replay_buffer = PrioritizedReplayBuffer(
            cfg.replay_size, cfg.replay_n_partitions,
            priority_func, cfg.alpha, cfg.beta_schedule, cfg.min_experiences)
        context.set_replay_buffer(cfg.update_batch_size, replay_buffer)

        AgentUpdater(cfg.steps_per_target_update, agent, target_agent)
        PolicyEvaluation_StateAction(action_value_function, action_values_from_returns, weight=1.0)

        optimizer = StochasticGradientDescent(
            agents=[agent], alg=cfg.optimizer_spec,
            lr_schedule=cfg.lr_schedule,
            opt_steps_per_update=cfg.opt_steps_per_update)

        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


config = Config(
    name="QLearning",
    get_updater=QLearning,

    build_policy=BuildEpsilonGreedyPolicy(),
    build_controller=BuildDuelingLstmController(),
    update_batch_size=32,
    optimizer_spec="adam",
    opt_steps_per_update=1,

    double=False,
    reverse_double=False,

    lr_schedule=1e-4,
    exploration_schedule="poly 1 10000 0.2",
    test_time_explore=0.01,

    gamma=1.0,
    lmbda=1.0,

    target_exploration_schedule=0.1,  # Epsilon for the target policy (the policy we are learning about).

    steps_per_target_update=1000,

    min_experiences=1000,

    replay_size=20000,
    replay_n_partitions=100,
    alpha=0.7,
    beta_schedule=1.0,
)


# RETRACE_CONFIG = QLEARNING_CONFIG.copy(
#     name="Retrace",
#     alg=Retrace,
#
#     controller=FeedforwardController(),
#
#     update_batch_size=16,
#     lr_schedule="0.001",
#     opt_steps_per_update=10,
#     steps_per_target_update=1000,
#     init_steps=1000,
#     lmbda=1.0,
#     exploration_schedule="poly 1.0 10000 0.1",
#     greedy_factor=10.0,
#     beta_schedule=0.0,
#     alpha=0.0,
#     test_time_explore=0.10,
#     normalize_imp_weights=False
# )
#
#
# TRQL_CONFIG = QLEARNING_CONFIG.copy(
#     name="TrustRegionQLearning",
#     alg=TrustRegionQLearning,
#     delta_schedule=0.01,
#     max_cg_steps=10,
#     max_line_search_steps=20,
# )
#
#
# PQL_CONFIG = QLEARNING_CONFIG.copy(
#     name="ProximalQLearning",
#     alg=ProximalQLearning,
#     opt_steps_per_update=10,
#     S=1,
#     epsilon=0.2,
# )
#
#
# DQN_CONFIG = QLEARNING_CONFIG.copy(
#     # From Nature paper
#
#     # Rewards are clipped: all negative rewards set to -1, all positive set to 1, 0 unchanged.
#     batch_size=32,
#
#     lr_schedule="0.01",
#     # lr_schedule="0.00025",
#
#     # annealed linearly from 1 to 0.1 over first million frames,
#     # fixed at 0.1 thereafter "
#     exploration_schedule="polynomial 1.0 10000 0.1 1",
#
#     replay_max_size=1e6,
#
#     # max number of frames/states: 10 million
#
#     test_time_explore=0.05,
#     # Once every 10000 frames
#     steps_per_target_update=10000,
#
#     gamma=0.99,
#
#     # 4 actions selected between each update
#     # RMS prop momentum: 0.95
#     # squared RMS prop gradient momentum: 0.95
#     # min squared gradient (RMSProp): 0.01
#     # exploration 1 to 0.1 over 1,000,000 frames
#     # total number of frames: 50,000,000, but because of frame skip, equivalent to 200,000,000 frames
# )
#
#
# FPS_CONFIG = QLEARNING_CONFIG.copy(
#     gamma=0.99,
#     update_batch_size=32,
#     batch_size=64,
#
#     # Exploration: annealed linearly from 1 to 0.1 over first million steps, fixed at 0.1 thereafter
#
#     # Replay max size: 1million *frames*
#
#     # They actually update every 4 *steps*, rather than every 4 experiences
#     samples_per_update=4,
# )
#
#
# DUELING_CONFIG = QLEARNING_CONFIG.copy(
#     max_grad_norm=10.0,
#     lr_schedule="6.25e-5",  # when prioritized experience replay was used
#     test_time_explore=0.001,  # fixed at this value...this might also be the training exploration, its not clear
# )
#
#
# DOUBLE_CONFIG = QLEARNING_CONFIG.copy(
#     double=True,
#     exploration_start=0.1,  # Not totally clear, but seems like they use the same scheme as DQN, but go from 1 to 0.01, instead of 1 to 0.1
#     test_time_explore=0.001,
#     # Target network update rate: once every 30,000 frames (DQN apparently does it once every 10,000 frames).
# )
#
