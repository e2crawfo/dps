import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps.utils import Parameterized, Param
from dps.utils.tf import (
    build_gradient_train_op, trainable_variables, scheduled_value_summaries)


class Updater(with_metaclass(abc.ABCMeta, Parameterized)):
    def __init__(self, env, scope=None, **kwargs):
        self.scope = scope
        self.env = env
        self._n_experiences = 0
        self._n_updates = 0

    @property
    def n_experiences(self):
        return self._n_experiences

    @property
    def n_updates(self):
        return self._n_updates

    def build_graph(self):
        with tf.name_scope(self.scope or self.__class__.__name__) as scope:
            self._scope = scope

            self._build_graph()

            self.scheduled_value_summaries_op = tf.summary.merge(scheduled_value_summaries())

            global_step = tf.train.get_or_create_global_step()
            self.inc_global_step_op = tf.assign_add(global_step, 1)
            global_step_input = tf.placeholder(tf.int64, ())
            assign_global_step = tf.assign(global_step, global_step_input)
            tf.get_default_session().run(assign_global_step, feed_dict={global_step_input: 0})

    @abc.abstractmethod
    def _build_graph(self):
        raise Exception("NotImplemented")

    def update(self, batch_size, collect_summaries):
        self._n_experiences += batch_size
        self._n_updates += 1

        sess = tf.get_default_session()

        scheduled_value_summaries = sess.run(self.scheduled_value_summaries_op)

        train, off_policy, train_record, off_policy_record = self._update(batch_size, collect_summaries)
        sess.run(self.inc_global_step_op)

        return (
            train + scheduled_value_summaries,
            off_policy + scheduled_value_summaries,
            train_record,
            off_policy_record
        )

    @abc.abstractmethod
    def _update(self, batch_size, collect_summaries=None):
        raise Exception("NotImplemented")

    def evaluate(self, batch_size, mode):
        summaries, record = self._evaluate(batch_size, mode)

        scheduled_value_summaries = \
            tf.get_default_session().run(self.scheduled_value_summaries_op)
        summaries += scheduled_value_summaries

        return summaries, record

    @abc.abstractmethod
    def _evaluate(self, batch_size, mode):
        assert mode in 'val test'.split()
        raise Exception("NotImplemented")

    def trainable_variables(self, for_opt):
        raise Exception("AbstractMethod")

    def save(self, session, filename):
        updater_variables = {v.name: v for v in self.trainable_variables(for_opt=False)}
        saver = tf.train.Saver(updater_variables)
        path = saver.save(tf.get_default_session(), filename)
        return path

    def restore(self, session, path):
        updater_variables = {v.name: v for v in self.trainable_variables(for_opt=False)}
        saver = tf.train.Saver(updater_variables)
        saver.restore(tf.get_default_session(), path)


class DifferentiableUpdater(Updater):
    """ Update parameters of a differentiable function `f` using gradient-based algorithm.

    Must be used in context of a default graph, session and config.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    f: An instance of ScopedFunction
        Accepts a tensor (input), returns a tensor (inference).

    """
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()
    l2_weight = Param(None)
    stopping_criteria = "loss,min"

    def __init__(self, env, f, **kwargs):
        assert hasattr(env, 'build'), (
            "Environments used with DifferentiableUpdater must possess "
            "a method called `build` which builds returns a dictionary of scalar tensors."
        )
        assert hasattr(env, 'make_feed_dict'), (
            "Environments used with DifferentiableUpdater must possess "
            "a method called `make_feed_dict` which return a feed dict "
            "to pass to the training step.")
        self.f = f

        super(DifferentiableUpdater, self).__init__(env, **kwargs)

    def trainable_variables(self, for_opt):
        return trainable_variables(self.f.scope, for_opt=for_opt)

    def _build_graph(self):
        self.recorded_tensors = self.env.build(self.f)
        self.loss = self.recorded_tensors['loss']

        tvars = self.trainable_variables(for_opt=True)
        if self.l2_weight is not None:
            self.loss += self.l2_weight * sum(tf.nn.l2_loss(v) for v in tvars if 'weights' in v.name)

        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.summary_op = tf.summary.merge(
            [tf.summary.scalar(name, t) for name, t in self.recorded_tensors.items()] +
            train_summaries)

    def _update(self, batch_size, collect_summaries):
        feed_dict = self.env.make_feed_dict(batch_size, 'train', False)

        train_summaries = b''

        sess = tf.get_default_session()
        if collect_summaries:
            train_summaries, _, record = sess.run(
                [self.summary_op, self.train_op, self.recorded_tensors], feed_dict=feed_dict)
        else:
            _, record = sess.run([self.train_op, self.recorded_tensors], feed_dict=feed_dict)

        return train_summaries, b'', record, {}

    def _evaluate(self, batch_size, mode):
        feed_dict = self.env.make_feed_dict(None, mode, True)

        sess = tf.get_default_session()
        summaries, record = sess.run(
            [self.summary_op, self.recorded_tensors], feed_dict=feed_dict)

        return summaries, record
