import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from dps.updater import Param
from dps.utils import build_scheduled_value
from dps.rl import (
    PolicyOptimization, GeneralizedAdvantageEstimator, BasicValueEstimator)
from dps.utils import build_gradient_train_op
from dps.policy import Categorical
from dps.rl.qLearning import PrioritizedReplayBuffer
from dps import cfg


class ACER(PolicyOptimization):
    """ Discrete version of Actor-Critic with Experience Replay. """

    optimizer_spec = Param()
    lr_schedule = Param()
    beta_schedule = Param(1.0)
    alpha = Param(0.0)

    def __init__(self, mu, policy, advantage_estimator=None, replay_buffer=None, **kwargs):
        self.mu = mu
        assert isinstance(mu.action_selection, Categorical)

        self.policy = policy

        if not advantage_estimator:
            advantage_estimator = GeneralizedAdvantageEstimator(BasicValueEstimator())
        self.advantage_estimator = advantage_estimator

        if replay_buffer is None:
            self.beta = build_scheduled_value(self.beta_schedule, 'beta')
            replay_buffer = PrioritizedReplayBuffer(self.replay_max_size, self.n_partitions, self.alpha, self.beta)

        super(ACER, self).__init__(**kwargs)

    def build_placeholders(self):
        super(ACER, self).build_placeholders()
        self.mu_utils = tf.placeholder(tf.float32, shape=(cfg.T, None, self.mu.n_actions), name="_mu_log_probs")
        self.mu_exploration = tf.placeholder(tf.float32, shape=(None,), name="_mu_exploration")

    def build_feeddict(self, rollouts):
        feed_dict = super(ACER, self).build_feed_dict(rollouts)
        feed_dict[self.mu_utils] = rollouts.utils
        feed_dict[self.mu_exploration] = rollouts.exploration

    def _build_graph(self, is_training, exploration):
        self.build_placeholders()

        self.policy.set_exploration(exploration)

        batch_size = tf.shape(self.obs)[1]
        (samples, utils), _ = dynamic_rnn(
            self.policy, self.obs, initial_state=self.policy.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)

        pi_log_probs = self.policy.build_log_prob_all(utils)
        pi_log_probs_selected_actions = tf.reduce_sum(pi_log_probs * self.actions, axis=-1, keep_dims=True)

        mu_exploration = tf.reshape(self.mu_exploration, (cfg.T, None, 1))
        mu_log_probs = self.mu.action_selection.log_prob_all(self.mu_utils, mu_exploration)
        mu_log_probs_selected_actions = tf.reduce_sum(mu_log_probs * self.actions, axis=-1, keep_dims=True)

        importance_ratio = tf.exp(pi_log_probs_selected_actions - mu_log_probs_selected_actions)
        rho = tf.minimum(importance_ratio, 1)

        main_objective_terms = tf.stop_gradient(rho * self.advantage) * pi_log_probs * self.mask
        main_objective = tf.reduce_mean(tf.reduce_sum(main_objective_terms, axis=0))

        rho_a = tf.exp(pi_log_probs - mu_log_probs)

        correction_terms = tf.stop_gradient(
            tf.exp(pi_log_probs) * tf.maximum((rho_a - self.c) / rho_a, 0.0) * self.advantage) * pi_log_probs
        correction_terms = tf.reduce_sum(correction_terms, axis=-1, keep_dims=True) * self.mask
        correction = tf.reduce_mean(tf.reduce_sum(correction_terms, axis=0))

        self.objective = main_objective + correction

        self.loss = -self.objective

        tvars = self.policy.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule)

        self.train_summary_op = tf.summary.merge(train_summaries)

        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("main_objective", main_objective),
            tf.summary.scalar("correction", correction),
            tf.summary.scalar("objective", self.objective),
            tf.summary.scalar("loss", -self.loss),
            tf.summary.scalar("reward_per_ep", self.reward_per_ep),
        ])

        self.recorded_values = [
            ('loss', -self.reward_per_ep),
            ('reward_per_ep', self.reward_per_ep),
        ]
