import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps.utils import (
    default_config, add_scaled_noise_to_gradients, inverse_time_decay)


class Updater(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        self._n_experiences = 0

    @property
    def n_experiences(self):
        return self._n_experiences

    def update(self, batch_size, summary_op=None):
        self._n_experiences += batch_size
        return self._update(batch_size, summary_op)

    @abc.abstractmethod
    def _update(self, batch_size, summary_op=None):
        raise NotImplementedError()

    def _build_train(self):
        """ Add ops to implement training. ``self.loss`` must already be defined. """
        with tf.name_scope('train'):
            start, decay_steps, decay_rate, staircase = self.lr_schedule
            lr = tf.train.exponential_decay(
                start, self.global_step, decay_steps, decay_rate, staircase=staircase)
            tf.summary.scalar('learning_rate', lr)

            self.optimizer = self.optimizer_class(lr)

            tvars = tf.trainable_variables()
            self.pure_gradients = tf.gradients(self.loss, tvars)

            if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0.0:
                self.clipped_gradients, _ = tf.clip_by_global_norm(self.pure_gradients, self.max_grad_norm)
            else:
                self.clipped_gradients = self.pure_gradients

            grads_and_vars = zip(self.clipped_gradients, tvars)
            if hasattr(self, 'noise_schedule'):
                start, decay_steps, decay_rate, staircase = self.noise_schedule
                noise = inverse_time_decay(
                    start, self.global_step, decay_steps, decay_rate, staircase=staircase, gamma=0.55)
                tf.summary.scalar('noise', noise)
                grads_and_vars = add_scaled_noise_to_gradients(grads_and_vars, noise)

            self.final_gradients = [g for g, v in grads_and_vars]
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

            # if default_config().debug:
            #     for grad, var in self.gradients:
            #         tf.histogram_summary(var.name, var)


class DifferentiableUpdater(Updater):
    """ Update parameters of a function ``f`` using vanilla gradient descent.

    The function must be differentiable to apply this updater.

    Should be used in the context of a default graph, default session and default config.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    f: callable
        Also needs to provide member functions ``build_feeddict`` and ``get_output``.
    global_step: Tensor
        Created by calling ``tf.contrib.learn.get_or_create_global_step``.

    """
    def __init__(self,
                 env,
                 f,
                 global_step,
                 optimizer_class,
                 lr_schedule,
                 noise_schedule,
                 max_grad_norm):

        super(DifferentiableUpdater, self).__init__()
        self.env = env
        self.f = f
        self.global_step = global_step
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule
        self.noise_schedule = noise_schedule
        self.max_grad_norm = max_grad_norm

        self._build_graph()

    def _update(self, batch_size, summary_op=None):

        env, loss = self.env, self.loss
        train_op, f, targets = self.train_op, self.f, self.target_placeholders
        sess = tf.get_default_session()

        train_x, train_y = env.train.next_batch(batch_size)
        if default_config().debug:
            print("x", train_x)
            print("y", train_y)

        feed_dict = f.build_feeddict(train_x)
        feed_dict[targets] = train_y

        if summary_op is not None:
            train_summary, train_loss, _ = sess.run([summary_op, loss, train_op], feed_dict=feed_dict)

            val_x, val_y = env.val.next_batch()
            val_feed_dict = f.build_feeddict(val_x)
            val_feed_dict[targets] = val_y

            val_summary, val_loss = sess.run([summary_op, loss], feed_dict=val_feed_dict)
            return train_summary, train_loss, val_summary, val_loss
        else:
            train_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            return train_loss

    def _build_graph(self):
        with tf.name_scope('loss'):
            self.loss, self.target_placeholders = self.env.loss(self.f.get_output())
            tf.summary.scalar('loss', self.loss)

        self._build_train()

    def checkpoint(self, path, saver):
        pass
