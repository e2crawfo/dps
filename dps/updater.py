import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps.utils import (
    build_gradient_train_op, Parameterized, Param,
    trainable_variables, scheduled_value_summaries)


class Updater(with_metaclass(abc.ABCMeta, Parameterized)):
    def __init__(self, env, **kwargs):
        self.scope = None
        self.env = env
        self._n_experiences = 0

    @property
    def n_experiences(self):
        return self._n_experiences

    def build_graph(self):
        with tf.variable_scope(self.__class__.__name__) as scope:
            self.scope = scope

            self.is_training = tf.Variable(False, trainable=False)
            self._set_is_training = tf.placeholder(tf.bool, ())
            self._assign_is_training = tf.assign(self.is_training, self._set_is_training)

            self._build_graph()

            self.scheduled_value_summaries_op = tf.summary.merge(scheduled_value_summaries())
            global_step = tf.contrib.framework.get_or_create_global_step()
            self.inc_global_step_op = tf.assign_add(global_step, 1)
            global_step_input = tf.placeholder(tf.int64, ())
            assign_global_step = tf.assign(global_step, global_step_input)
            tf.get_default_session().run(assign_global_step, feed_dict={global_step_input: 0})

    @abc.abstractmethod
    def _build_graph(self):
        raise Exception("NotImplemented")

    def update(self, batch_size, collect_summaries):
        self._n_experiences += batch_size
        self.set_is_training(True)

        scheduled_value_summaries = \
            tf.get_default_session().run(self.scheduled_value_summaries_op)

        train, replay = self._update(batch_size, collect_summaries)
        tf.get_default_session().run(self.inc_global_step_op)
        return train + scheduled_value_summaries, replay

    @abc.abstractmethod
    def _update(self, batch_size, collect_summaries=None):
        raise Exception("NotImplemented")

    def evaluate(self, batch_size):
        self.set_is_training(False)

        loss, summaries, record = self._evaluate(batch_size)

        scheduled_value_summaries = \
            tf.get_default_session().run(self.scheduled_value_summaries_op)
        summaries += scheduled_value_summaries

        return loss, summaries, record

    @abc.abstractmethod
    def _evaluate(self, batch_size, mode):
        raise Exception("NotImplemented")

    def set_is_training(self, is_training):
        tf.get_default_session().run(
            self._assign_is_training, feed_dict={self._set_is_training: is_training})

    def save(self, session, filename):
        if self.scope is None:
            raise Exception(
                "Cannot save variables for an Updater that "
                "has not had its `build_graph` method called.")
        updater_variables = trainable_variables(self.scope.name)
        saver = tf.train.Saver(updater_variables)
        path = saver.save(tf.get_default_session(), filename)
        return path

    def restore(self, session, path):
        if self.scope is None:
            raise Exception(
                "Cannot save variables for an Updater that has "
                "not had its `build_graph` method called.")
        updater_variables = trainable_variables(self.scope.name)
        saver = tf.train.Saver(updater_variables)
        saver.restore(tf.get_default_session(), path)


class DifferentiableUpdater(Updater):
    """ Update parameters of a differentiable function `f` using gradient-based algorithm.

    Must be used in context of a default graph, session and config.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    f: (differentiable) callable
        Accepts a tensor (input), returns a tensor (inference).

    """
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    def __init__(self, env, f, **kwargs):
        assert hasattr(env, 'build_loss'), (
            "Environments used with DifferentiableUpdater must possess "
            "a method called `build_loss` which builds a differentiable "
            "loss function.")
        self.f = f
        self.obs_shape = env.obs_shape
        self.actions_dim = env.actions_dim

        super(DifferentiableUpdater, self).__init__(env)

    def _build_graph(self):
        self.x_ph = tf.placeholder(tf.float32, (None,) + self.obs_shape)
        self.target_ph = tf.placeholder(tf.float32, (None, self.actions_dim))
        self.output = self.f(self.x_ph)
        self.loss = tf.reduce_mean(self.env.build_loss(self.output, self.target_ph))
        self.reward = tf.reduce_mean(self.env.build_reward(self.output, self.target_ph))
        self.mean_value = tf.reduce_mean(self.output)

        tvars = trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.train_summary_op = tf.summary.merge(train_summaries)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss_per_ep", self.loss),
            tf.summary.scalar("reward_per_ep", self.reward),
            tf.summary.scalar("mean_value", self.mean_value)
        ] + train_summaries)

    def _update(self, batch_size, collect_summaries):
        sess = tf.get_default_session()
        train_x, train_y = self.env.next_batch(batch_size, mode='train')

        feed_dict = {
            self.x_ph: train_x,
            self.target_ph: train_y
        }

        sess = tf.get_default_session()
        if collect_summaries:
            train_summaries, _ = sess.run([self.train_summary_op, self.train_op], feed_dict=feed_dict)
            return train_summaries, b''
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            return b'', b''

    def _evaluate(self, batch_size, mode):
        x, y = self.env.next_batch(None, mode=mode)

        feed_dict = {
            self.x_ph: x,
            self.target_ph: y
        }

        sess = tf.get_default_session()
        summaries, loss, reward, mean_value = sess.run(
            [self.summary_op, self.loss, self.reward, self.mean_value], feed_dict=feed_dict)
        return loss, summaries, {"reward": reward, "mean_value": mean_value}
