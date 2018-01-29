import tensorflow as tf

from dps import updater
from dps.utils import Param
from dps.utils.tf import build_gradient_train_op, trainable_variables


class ProgramGenerationUpdater(updater.Updater):
    def __init__(self, env, scope, **kwargs):
        super(ProgramGenerationUpdater, self).__init__(env, **kwargs)

    def set_is_training(self, is_training):
        tf.get_default_session().run(
            self._assign_is_training, feed_dict={self._set_is_training: is_training})

    def trainable_variables(self, for_opt):
        return trainable_variables(self.f.scope, for_opt=for_opt)

    def _build_graph(self):
        self.is_training = tf.Variable(False, trainable=False, name="is_training")
        self._set_is_training = tf.placeholder(tf.bool, ())
        self._assign_is_training = tf.assign(self.is_training, self._set_is_training)

        self.x_ph = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="x_ph")
        self.target_ph = tf.placeholder(tf.float32, (None,) + self.action_shape, name="target_ph")
        self.output = self.f(self.x_ph, self.action_shape, self.is_training)
        self.loss = tf.reduce_mean(self.env.build_loss(self.output, self.target_ph))

        self.recorded_tensors = [
            tf.reduce_mean(getattr(self.env, 'build_' + name)(self.output, self.target_ph))
            for name in self.env.recorded_names
        ]

        tvars = self.trainable_variables(for_opt=True)
        if self.l2_weight is not None:
            self.loss += self.l2_weight * sum(tf.nn.l2_loss(v) for v in tvars if 'weights' in v.name)

        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.summary_op = tf.summary.merge(
            [tf.summary.scalar("loss_per_ep", self.loss)] +
            [tf.summary.scalar(name, t) for name, t in zip(self.env.recorded_names, self.recorded_tensors)] +
            train_summaries)

    def _update(self, batch_size, collect_summaries):
        self.set_is_training(True)
        x, y = self.env.next_batch(batch_size, mode='train')

        sess = tf.get_default_session()
        feed_dict = {
            self.x_ph: x,
            self.target_ph: y
        }

        sess = tf.get_default_session()
        if collect_summaries:
            train_summaries, _, *recorded_values = sess.run(
                [self.summary_op, self.train_op] + self.recorded_tensors, feed_dict=feed_dict)
            return train_summaries, b'', dict(zip(self.env.recorded_names, recorded_values)), {}
        else:
            _, *recorded_values = sess.run([self.train_op] + self.recorded_tensors, feed_dict=feed_dict)
            return b'', b'', dict(zip(self.env.recorded_names, recorded_values)), {}

    def _evaluate(self, batch_size, mode):
        self.set_is_training(False)

        x, y = self.env.next_batch(None, mode)

        feed_dict = {
            self.x_ph: x,
            self.target_ph: y
        }

        sess = tf.get_default_session()
        summaries, loss, *recorded_values = sess.run(
            [self.summary_op, self.loss] + self.recorded_tensors, feed_dict=feed_dict)

        record = {"loss": loss}
        record.update(zip(self.env.recorded_names, recorded_values))

        return summaries, record
