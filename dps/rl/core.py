import numpy as np
import tensorflow as tf

from dps import cfg
from dps.utils import build_scheduled_value, Param, Parameterized
from dps.updater import Updater


def shift_zero_fill(a, n, axis=0):
    shifted = np.roll(a, n, axis=axis)
    shifted[:n, ...] = 0.0
    return shifted


def rl_render_hook(updater):
    if hasattr(updater, 'policy'):
        render_rollouts = getattr(cfg, 'render_rollouts', None)
        exploration = tf.get_default_session().run(updater.exploration)
        updater.env.visualize(
            policy=updater.policy, exploration=exploration,
            n_rollouts=4, T=cfg.T, mode='train',
            render_rollouts=render_rollouts)
    else:
        print("Not rendering.")


def episodic_mean(v, name=None):
    """ Sum across time steps, take mean across episodes """
    return tf.reduce_mean(tf.reduce_sum(v, axis=0), name=name)


class RLUpdater(Updater):
    """ Update parameters of objects (mainly policies and value functions)
        based on sequences of interactions between a behaviour policy and
        an environment.

    Must be used in context of a default graph, session and config.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    policy: callable object
        Needs to provide member functions ``build_feeddict`` and ``get_output``.
    learners: Learner instance or list thereof
        Objects that learn from the trajectories sampled from interaction
        between `env` and `policy`.

    """
    exploration_schedule = Param()
    test_time_explore = Param()

    def __init__(self, env, policy, learners=None, **kwargs):
        self.env = env
        self.policy = policy

        learners = learners or []
        try:
            self.learners = list(learners)
        except:
            self.learners = [learners]

        self.obs_shape = env.obs_shape
        self.n_actions = env.n_actions

        super(RLUpdater, self).__init__(env)

    def _build_graph(self):
        training_exploration = build_scheduled_value(self.exploration_schedule)
        if isinstance(self.test_time_explore, str) or self.test_time_explore >= 0:
            testing_exploration = build_scheduled_value(self.test_time_explore)
            self.exploration = tf.cond(self.is_training, lambda: training_exploration, lambda: testing_exploration)
        else:
            self.exploration = training_exploration
        tf.summary.scalar('exploration', self.exploration, collections=['scheduled_value_summaries'])

        self.policy.set_exploration(self.exploration)

        # self.policy.build_update()

        for learner in self.learners:
            learner.build_graph(self.is_training, self.exploration)

        self.reward = tf.placeholder(tf.float32, (None, None, 1), "_bp_reward")
        self.reward_per_ep = episodic_mean(self.reward, name="_bp_reward_per_ep")

    def _update(self, batch_size, collect_summaries):
        rollouts = self.env.do_rollouts(self.policy, batch_size, mode='train')
        rollouts['mask'] = (1-shift_zero_fill(rollouts.done, 1)).astype('f')

        summaries = (b'').join(
            learner.update(rollouts, collect_summaries)
            for learner in self.learners)
        return summaries

    def _evaluate(self, batch_size, mode):
        """ Return list of tf summaries and a dictionary of values to be displayed. """
        rollouts = self.env.do_rollouts(self.policy, batch_size, mode=mode)
        rollouts['mask'] = 1-shift_zero_fill(rollouts.done, 1)

        summaries = b''
        record = {}

        # Give precedence to learners that occur earlier.
        for learner in self.learners[::-1]:
            s, r = learner.evaluate(rollouts)
            summaries += s
            record.update(r)
        assert 'loss' in record

        sess = tf.get_default_session()
        feed_dict = {self.reward: rollouts.r}
        record['behaviour_policy_reward_per_ep'] = sess.run(self.reward_per_ep, feed_dict=feed_dict)

        return summaries, record


class ReinforcementLearner(Parameterized):
    """ Anything that learns from batches of trajectories.

    Doesn't necessarily need to learn about policies; can also learn about e.g. value functions.

    """
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", self.__class__.__name__)
        super(ReinforcementLearner, self).__init__(**kwargs)

    def _build_graph(self, is_training, exploration):
        raise Exception("NotImplemented")

    def build_graph(self, is_training, exploration):
        with tf.name_scope(self.name):
            self._build_graph(is_training, exploration)

    def update(self, rollouts, collect_summaries):
        pass

    def evaluate(self, rollouts):
        pass


class PolicyOptimization(ReinforcementLearner):
    def compute_advantage(self, rollouts):
        advantage = self.advantage_estimator.estimate(rollouts)
        if cfg.standardize_advantage:
            _advantage = advantage[rollouts.mask.nonzero()]
            _advantage = _advantage - _advantage.mean()
            adv_std = _advantage.std()
            if adv_std > 1e-6:
                _advantage /= adv_std
            advantage[rollouts.mask.nonzero()] = _advantage

        return advantage

    def build_placeholders(self):
        self.obs = tf.placeholder(
            tf.float32, shape=(cfg.T, None) + self.policy.obs_shape, name="_obs")
        self.actions = tf.placeholder(
            tf.float32, shape=(cfg.T, None, self.policy.n_actions), name="_actions")
        self.advantage = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_advantage")
        self.rewards = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_rewards")
        self.mask = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_mask")
        self.reward_per_ep = episodic_mean(self.rewards, name="_reward_per_ep")

    def update(self, rollouts, collect_summaries):
        advantage = self.compute_advantage(rollouts)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.advantage: advantage,
            self.mask: rollouts.mask
        }

        sess = tf.get_default_session()
        if collect_summaries:
            train_summaries, _ = sess.run(
                [self.train_summary_op, self.train_op], feed_dict=feed_dict)
            return train_summaries
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            return b''

    def evaluate(self, rollouts):
        advantage = self.compute_advantage(rollouts)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.advantage: advantage,
            self.mask: rollouts.mask
        }

        sess = tf.get_default_session()

        eval_summaries, *values = (
            sess.run(
                [self.eval_summary_op] + [v for _, v in self.recorded_values],
                feed_dict=feed_dict))

        record = {k: v for v, (k, _) in zip(values, self.recorded_values)}
        return eval_summaries, record
