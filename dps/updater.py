import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps import cfg
from dps.utils import (
    add_scaled_noise_to_gradients, build_scheduled_value, build_optimizer)


class Param(object):
    pass


class Updater(with_metaclass(abc.ABCMeta, object)):
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()
    gamma = Param()
    l2_norm_penalty = Param()

    def __init__(self, env, **kwargs):
        self.env = env
        self._n_experiences = 0
        self._resolve_params(kwargs)

        self.build_graph()

    def _resolve_params(self, kwargs):
        for p in self.params:
            value = kwargs.get(p)
            if value is None:
                value = getattr(cfg, p)
            setattr(self, p, value)

    @property
    def params(self):
        return [p for p in dir(self) if p != 'params' and isinstance(getattr(self, p), Param)]

    @property
    def n_experiences(self):
        return self._n_experiences

    def update(self, batch_size, summary_op=None):
        self._n_experiences += batch_size
        return self._update(batch_size, summary_op)

    @abc.abstractmethod
    def _update(self, batch_size, summary_op=None):
        raise Exception()

    @abc.abstractmethod
    def build_graph(self):
        raise Exception()

    def _build_optimizer(self):
        """ Helper method that can be called by ``build_graph``.
            Requires that `self.loss` be set to a Tensor.
        """
        lr = build_scheduled_value(self.lr_schedule, 'learning_rate')
        self.optimizer = build_optimizer(self.optimizer_spec, lr)

        # We only optimize trainable variables from the policy
        g = tf.get_default_graph()
        tvars = g.get_collection('trainable_variables', scope=self.policy.scope.name)
        pure_gradients = tf.gradients(self.loss, tvars)

        clipped_gradients = pure_gradients
        if hasattr(self, 'max_grad_norm') and self.max_grad_norm is not None and self.max_grad_norm > 0.0:
            clipped_gradients, _ = tf.clip_by_global_norm(pure_gradients, self.max_grad_norm)

        global_step = tf.contrib.framework.get_or_create_global_step()

        noisy_gradients = clipped_gradients
        if hasattr(self, 'noise_schedule') and self.noise_schedule is not None:
            grads_and_vars = zip(clipped_gradients, tvars)
            noise = build_scheduled_value(self.noise_schedule, 'gradient_noise')
            noisy_gradients = add_scaled_noise_to_gradients(grads_and_vars, noise)

        grads_and_vars = list(zip(noisy_gradients, tvars))
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # for grad, var in self.gradients:
        #     tf.histogram_summary(var.name, var)

        tf.summary.scalar('grad_norm_pure', tf.global_norm(pure_gradients))
        tf.summary.scalar('grad_norm_clipped', tf.global_norm(clipped_gradients))
        tf.summary.scalar('grad_norm_clipped_and_noisy', tf.global_norm(noisy_gradients))


class ReinforcementLearningUpdater(Updater):
    """ Update parameters of a policy using reinforcement learning.

    Must be used in context of a default graph, session and config.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    policy: callable object
        Needs to provide member functions ``build_feeddict`` and ``get_output``.

    """
    def __init__(self, env, policy, **kwargs):
        self.policy = policy
        self.obs_shape = env.observation_space.shape[1:]
        self.n_actions = env.action_space.shape[1]

        super(ReinforcementLearningUpdater, self).__init__(env, **kwargs)

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear_buffers(self):
        self.obs_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

    def remember(self, obs, action, reward, behaviour_policy=None):
        """ Supply the RL algorithm with a unit of experience. """
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)


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
    def __init__(self, env, f, **kwargs):
        assert hasattr(env, 'build_loss'), (
            "Environments used with DifferentiableUpdater must possess "
            "a method called `build_loss` which builds a differentiable loss function.")
        self.policy = f
        self.obs_shape = env.observation_space.shape[1:]
        self.n_actions = env.action_space.shape[1]
        super(DifferentiableUpdater, self).__init__(env, **kwargs)

    def _update(self, batch_size, summary_op=None):
        sess = tf.get_default_session()

        train_x, train_y = self.env.train.next_batch(batch_size)

        feed_dict = {
            self.inp_ph: train_x,
            self.target_ph: train_y,
            self.is_training: True}

        if summary_op is not None:
            train_summary, train_loss, _ = sess.run([summary_op, self.loss, self.train_op], feed_dict=feed_dict)

            val_x, val_y = self.env.val.next_batch()
            feed_dict = {
                self.inp_ph: val_x,
                self.target_ph: val_y,
                self.is_training: False}

            val_summary, val_loss = sess.run([summary_op, self.loss], feed_dict=feed_dict)
            return train_summary, train_loss, val_summary, val_loss
        else:
            train_loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            return train_loss

    def build_graph(self):
        with tf.name_scope("updater"):
            self.inp_ph = tf.placeholder(tf.float32, (None,) + self.obs_shape)
            self.target_ph = tf.placeholder(tf.float32, (None, None))
            self.output = self.policy(self.inp_ph)
            self.loss = tf.reduce_mean(self.env.build_loss(self.output, self.target_ph))
            self._build_optimizer()

        with tf.name_scope("performance"):
            tf.summary.scalar("surrogate_loss", self.loss)
            tf.summary.scalar("reward_per_ep", -self.loss)
