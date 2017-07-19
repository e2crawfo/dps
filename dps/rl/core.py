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

        self.obs_shape = env.observation_space.shape[1:]
        self.n_actions = env.action_space.shape[1]

        super(RLUpdater, self).__init__(env, **kwargs)

    def _build_graph(self):
        training_exploration = build_scheduled_value(self.exploration_schedule, 'exploration')

        if self.test_time_explore >= 0:
            testing_exploration = tf.constant(self.test_time_explore, tf.float32, name='testing_exploration')
            self.exploration = tf.cond(self.is_training, lambda: training_exploration, lambda: testing_exploration)
        else:
            self.exploration = training_exploration

        self.policy.set_exploration(self.exploration)

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
        rollouts = self.env.do_rollouts(self.policy, batch_size, mode='train')

        summaries = (b'').join(
            learner.update(rollouts, collect_summaries)
            for learner in self.learners)
        return summaries

    def evaluate(self, batch_size, mode):
        """ Return list of tf summaries and a dictionary of values to be displayed. """
        assert mode in 'train_eval val'.split()
        self.set_is_training(mode == 'train_eval')

        rollouts = self.env.do_rollouts(self.policy, batch_size, mode=mode)

        summaries = b''
        record = {}
        loss = None

        for learner in self.learners:
            l, s, r = learner.evaluate(rollouts)
            if loss is None:
                loss = l
            summaries += s
            overlap = record.keys() & r.keys()
            if overlap:
                raise Exception(str(overlap))
            record.update(r)

        sess = tf.get_default_session()
        feed_dict = {self.reward: rollouts.r}
        _summaries, reward_per_ep = (
            sess.run([self.eval_summary_op, self.reward_per_ep], feed_dict=feed_dict))
        summaries += _summaries
        record['behaviour_policy_reward_per_ep'] = reward_per_ep

        return loss, summaries, record


class ReinforcementLearner(Parameterized):
    """ Anything that learns from batches of trajectories.

    Doesn't necessarily need to learn about policies; can also learn about e.g. value functions.

    """
    def update(self, rollouts, collect_summaries):
        pass

    def evaluate(self, rollouts):
        pass
