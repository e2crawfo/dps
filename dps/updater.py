import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps.utils import (
    default_config, add_scaled_noise_to_gradients,
    adj_inverse_time_decay, build_scheduled_value, build_optimizer)


class Updater(with_metaclass(abc.ABCMeta, object)):

    def __init__(self, env):
        self.env = env
        self._n_experiences = 0

        self.build_graph()

    @property
    def stage(self):
        return 0

    @property
    def n_experiences(self):
        return self._n_experiences

    def update(self, batch_size, summary_op=None):
        self._n_experiences += batch_size
        return self._update(batch_size, summary_op)

    @abc.abstractmethod
    def _update(self, batch_size, summary_op=None):
        raise NotImplementedError()

    def build_graph(self):
        self._build_graph()

    @abc.abstractmethod
    def _build_graph(self):
        raise NotImplementedError()

    def _build_train(self):
        """ Add ops to implement training. ``self.loss`` must already be defined. """

        tf.summary.scalar('loss', self.loss)

        lr = build_scheduled_value(self.lr_schedule, 'learning_rate')

        self.optimizer = build_optimizer(self.optimizer_spec, lr)

        tvars = tf.trainable_variables()
        self.pure_gradients = tf.gradients(self.loss, tvars)
        tf.summary.scalar('gradient_norm_pure', tf.global_norm(self.pure_gradients))

        if hasattr(self, 'max_grad_norm') and self.max_grad_norm is not None and self.max_grad_norm > 0.0:
            self.clipped_gradients, _ = tf.clip_by_global_norm(self.pure_gradients, self.max_grad_norm)
        else:
            self.clipped_gradients = self.pure_gradients

        tf.summary.scalar('gradient_norm_clipped', tf.global_norm(self.clipped_gradients))

        global_step = tf.contrib.framework.get_or_create_global_step()

        if hasattr(self, 'noise_schedule') and self.noise_schedule is not None:
            grads_and_vars = zip(self.clipped_gradients, tvars)
            start, decay_steps, decay_rate, staircase = self.noise_schedule
            noise = adj_inverse_time_decay(
                start, global_step, decay_steps, decay_rate, staircase=staircase, gamma=0.55)
            tf.summary.scalar('gradient_noise_amount', noise)
            self.noisy_gradients = add_scaled_noise_to_gradients(grads_and_vars, noise)
        else:
            self.noisy_gradients = self.clipped_gradients
        tf.summary.scalar('gradient_norm_clipped_and_noisy', tf.global_norm(self.noisy_gradients))

        grads_and_vars = list(zip(self.noisy_gradients, tvars))
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # if default_config().debug:
        #     for grad, var in self.gradients:
        #         tf.histogram_summary(var.name, var)

    def save(self, path, step=None):
        g = tf.get_default_graph()
        tvars = g.get_collection('trainable_variables')
        saver = tf.train.Saver(tvars)
        return saver.save(tf.get_default_session(), path, step)

    def restore(self, path):
        g = tf.get_default_graph()
        tvars = g.get_collection('trainable_variables')
        saver = tf.train.Saver(tvars)
        saver.restore(tf.get_default_session(), path)


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

    """
    def __init__(self,
                 env,
                 f,
                 optimizer_spec,
                 lr_schedule,
                 noise_schedule,
                 max_grad_norm):

        self.f = f
        self.optimizer_spec = optimizer_spec
        self.lr_schedule = lr_schedule
        self.noise_schedule = noise_schedule
        self.max_grad_norm = max_grad_norm

        super(DifferentiableUpdater, self).__init__(env)

    def _update(self, batch_size, summary_op=None):
        sess = tf.get_default_session()

        train_x, train_y = self.env.train.next_batch(batch_size)
        if default_config().debug:
            print("x", train_x)
            print("y", train_y)

        feed_dict = self.f.build_feeddict(train_x)
        feed_dict[self.target_placeholders] = train_y
        feed_dict[self.is_training] = True

        if summary_op is not None:
            train_summary, train_loss, _ = sess.run([summary_op, self.loss, self.train_op], feed_dict=feed_dict)

            val_x, val_y = self.env.val.next_batch()
            val_feed_dict = self.f.build_feeddict(val_x)
            val_feed_dict[self.target_placeholders] = val_y
            val_feed_dict[self.is_training] = False

            val_summary, val_loss = sess.run([summary_op, self.loss], feed_dict=val_feed_dict)
            return train_summary, train_loss, val_summary, val_loss
        else:
            train_loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            return train_loss

    def _build_graph(self):
        with tf.name_scope("train"):
            loss, self.target_placeholders = self.env.build_loss(self.f.get_output())
            self.loss = tf.reduce_mean(loss)

            tf.summary.scalar("reward_per_ep", -self.loss)

            self._build_train()


class ReinforcementLearningUpdater(Updater):
    """ Update parameters of a policy using reinforcement learning.

    Should be used in the context of both a default session, default graph and default context.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    policy: callable object
        Needs to provide member functions ``build_feeddict`` and ``get_output``.

    """
    def __init__(self,
                 env,
                 policy,
                 optimizer_spec,
                 lr_schedule,
                 noise_schedule,
                 max_grad_norm,
                 gamma,
                 l2_norm_penalty):

        assert policy.action_selection.can_sample, (
            "Cannot sample when using action selection method {}".format(policy.action_selection))
        self.policy = policy
        self.optimizer_spec = optimizer_spec
        self.lr_schedule = lr_schedule
        self.noise_schedule = noise_schedule
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.l2_norm_penalty = l2_norm_penalty

        self.obs_dim = env.observation_space.shape[1]
        self.n_actions = env.action_space.shape[1]

        super(ReinforcementLearningUpdater, self).__init__(env)

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def remember(self, obs, action, reward, behaviour_policy=None):
        """ Supply the RL algorithm with a unit of experience.

        If behaviour_policy==None, assumes that data was generated by self.policy.

        """
        # Note to self: Make it so every argument can be a batch.
        if not self.policy:
            raise ValueError("Policy has not been set using ``set_policy``.")
