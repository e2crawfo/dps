import numpy as np
import tensorflow as tf

from dps import cfg
from dps.utils import build_scheduled_value, Param, Parameterized
from dps.updater import Updater


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


class RolloutBatch(object):
    """ Assumes components are stored with shape (time, batch_size, dimension) """

    def __init__(self, o=None, a=None, r=None, lp=None):
        self._o = list(o or [])
        self._a = list(a or [])
        self._r = list(r or [])
        self._lp = list(lp or [])

    def append(self, o, a, r, lp):
        self._o.append(o)
        self._a.append(a)
        self._r.append(r)
        self._lp.append(lp)

    @property
    def o(self):
        return np.array(self._o)

    @property
    def a(self):
        return np.array(self._a)

    @property
    def r(self):
        return np.array(self._r)

    @property
    def p(self):
        return np.array(self._lp)

    def clear(self):
        self._o = []
        self._a = []
        self._r = []
        self._lp = []

    def T(self):
        return len(self._o)

    def batch_size(self):
        return self._o[0].shape[0]

    def obs_shape(self):
        return self._o[0].shape[1:]

    def action_shape(self):
        return self._a[0].shape[1:]

    def n_rewards(self):
        return self._r[0].shape[1]


def episodic_mean(v, name=None):
    """ Sum across time steps, take mean across episodes """
    return tf.reduce_mean(tf.reduce_sum(v, axis=0), name=name)


class ReinforcementLearningUpdater(Updater):
    """ Update parameters of objects (mainly policies and value functions)
        based on sequences of interactions between a behaviour policy and an environment.

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

        self.obs_shape = env.observation_space.shape[1:]
        self.n_actions = env.action_space.shape[1]

        self.rollouts = RolloutBatch()

        super(ReinforcementLearningUpdater, self).__init__(env, **kwargs)

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear_buffers(self):
        self.rollouts.clear()

    def remember(self, obs, action, reward, log_prob):
        """ Supply the RL algorithm with a unit of experience. """
        self.rollouts.append(obs, action, reward, log_prob)

    def set_is_training(self, is_training):
        tf.get_default_session().run(self._assign_is_training, feed_dict={self._set_is_training: is_training})

    def _build_graph(self):
        self.is_training = tf.Variable(False, trainable=False)
        self._set_is_training = tf.placeholder(tf.bool, ())
        self._assign_is_training = tf.assign(self.is_training, self._set_is_training)

        training_exploration = build_scheduled_value(self.exploration_schedule, 'exploration')

        if self.test_time_explore >= 0:
            testing_exploration = tf.constant(self.test_time_explore, tf.float32, name='testing_exploration')
            self.exploration = tf.cond(self.is_training, lambda: training_exploration, lambda: testing_exploration)
        else:
            self.exploration = training_exploration

        # self.policy.build_update()
        for learner in self.learners:
            learner.build_graph(self.is_training, self.exploration)

        self.reward = tf.placeholder(tf.float32, (None, None, 1))
        self.reward_per_ep = episodic_mean(self.reward)

        with tf.name_scope("eval"):
            self.eval_summary_op = tf.summary.merge([
                tf.summary.scalar("reward_per_ep", self.reward_per_ep)
            ])

    def _update(self, batch_size, collect_summaries):
        self.set_is_training(True)
        self.clear_buffers()
        self.env.do_rollouts(self, self.policy, batch_size, mode='train')

        summaries = (b'').join(
            learner.update(self.rollouts, collect_summaries)
            for learner in self.learners)
        return summaries

    def evaluate(self, batch_size, mode):
        """ Return list of tf summaries and a dictionary of values to be displayed. """
        assert mode in 'train_eval val'.split()
        self.set_is_training(mode == 'train_eval')

        self.clear_buffers()

        self.env.do_rollouts(self, self.policy, batch_size, mode=mode)

        summaries = b''
        record = {}
        for learner in self.learners:
            s, r = learner.evaluate(self.rollouts)
            summaries += s
            overlap = record.keys() & r.keys()
            if overlap:
                raise Exception(str(overlap))
            record.update(r)

        sess = tf.get_default_session()
        feed_dict = {self.reward: self.rollouts.r}
        _summaries, reward_per_ep = (
            sess.run([self.eval_summary_op, self.reward_per_ep], feed_dict=feed_dict))
        summaries += _summaries
        record['behaviour_policy_reward_per_ep'] = reward_per_ep

        return -reward_per_ep, summaries, record


class ReinforcementLearner(Parameterized):
    """ Anything that learns from batches of trajectories.

    Doesn't necessarily need to learn about policies; can also learn about e.g. value functions.

    """
    def update(self, rollouts, collect_summaries):
        pass

    def evaluate(self, rollouts):
        pass
