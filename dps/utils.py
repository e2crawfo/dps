import subprocess as sp
from pprint import pformat
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.python.ops import random_ops, math_ops
from tensorflow.python.framework import ops, constant_op


# class EarlyStopHook(SessionRunHook):
class EarlyStopHook(object):
    def __init__(self, patience, n=1, name=None):
        self.patience = patience
        self.name = name

        self._early_stopped = 0
        self._best_value_step = None
        self._best_value = None

        self._step = 0

    @property
    def early_stopped(self):
        """Returns True if this monitor caused an early stop."""
        return self._early_stopped

    @property
    def best_step(self):
        """Returns the step at which the best early stopping metric was found."""
        return self._best_value_step

    @property
    def best_value(self):
        """Returns the best early stopping metric value found so far."""
        return self._best_value

    def check(self, validation_loss):
        new_best = self._best_value is None or validation_loss < self._best_value
        if new_best:
            self._best_value = validation_loss
            self._best_value_step = self._step

        stop = self._step - self._best_value_step > self.patience
        if stop:
            print("Stopping. Best step: {} with loss = {}." .format(
                  self._best_value_step, self._best_value))
            self._early_stopped = True
        self._step += 1
        return new_best, stop


def restart_tensorboard(logdir):
    print("Killing old tensorboard process...")
    try:
        sp.run("fuser 6006/tcp -k".split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        print("Killing tensorboard failed:")
        print(e.output)
    print("Restarting tensorboard process...")
    sp.Popen("tensorboard --logdir={}".format(logdir).split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    print("Done restarting tensorboard.")


def training(loss, global_step, config=None):
    config = config or default_config()
    with tf.name_scope('train'):
        start, decay_steps, decay_rate, staircase = config.lr_schedule
        lr = tf.train.exponential_decay(
            start, global_step, decay_steps, decay_rate, staircase=staircase)
        tf.summary.scalar('learning_rate', lr)

        opt = config.optimizer_class(lr)

        tvars = tf.trainable_variables()
        if hasattr(config, 'max_grad_norm') and config.max_grad_norm > 0.0:
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_grad_norm)
        else:
            grads = tf.gradients(loss, tvars)

        grads_and_vars = zip(grads, tvars)
        if hasattr(config, 'noise_schedule'):
            start, decay_steps, decay_rate, staircase = config.noise_schedule
            noise = inverse_time_decay(
                start, global_step, decay_steps, decay_rate, staircase=staircase, gamma=0.55)
            tf.summary.scalar('noise', noise)
            grads_and_vars = add_scaled_noise_to_gradients(grads_and_vars, noise)

        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op


def MSE(outputs, targets):
    return tf.reduce_mean(tf.square(tf.subtract(targets, outputs)))


def add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
    """Taken from tensorflow.

    Adds scaled noise from a 0-mean normal distribution to gradients.

    """
    gradients, variables = zip(*grads_and_vars)
    noisy_gradients = []
    for gradient in gradients:
        if gradient is None:
            noisy_gradients.append(None)
            continue
        if isinstance(gradient, ops.IndexedSlices):
            gradient_shape = gradient.dense_shape
        else:
            gradient_shape = gradient.get_shape()
        noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
        noisy_gradients.append(gradient + noise)
    return list(zip(noisy_gradients, variables))


def inverse_time_decay(initial, global_step, decay_steps, decay_rate, gamma,
                       staircase=False, name=None):
    """Applies inverse time decay to the initial learning rate.

    Adapted from tf.train.inverse_time_decay (added `gamma` arg.)

    The function returns the decayed learning rate.  It is computed as:

    ```python
    decayed_value = initial / (1 + decay_rate * t/decay_steps)
    ```

    Args:
      initial: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      global_step: A Python number.
        Global step to use for the decay computation.  Must not be negative.
      decay_steps: How often to apply decay.
      decay_rate: A Python number.  The decay rate.
      staircase: Whether to apply decay in a discrete staircase, as opposed to
        continuous, fashion.
      gamma: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The power to raise output to.
      name: String.  Optional name of the operation.  Defaults to
        'InverseTimeDecay'.

    Returns:
      A scalar `Tensor` of the same type as `initial`.  The decayed
      learning rate.

    Raises:
      ValueError: if `global_step` is not supplied.

    """
    if global_step is None:
        raise ValueError("global_step is required for inverse_time_decay.")

    with ops.name_scope(name, "InverseTimeDecay",
                        [initial, global_step, decay_rate]) as name:
        initial = ops.convert_to_tensor(initial, name="initial")
        dtype = initial.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        decay_rate = math_ops.cast(decay_rate, dtype)
        p = global_step / decay_steps
        if staircase:
            p = math_ops.floor(p)
        const = math_ops.cast(constant_op.constant(1), initial.dtype)
        denom = math_ops.add(const, math_ops.multiply(decay_rate, p))
        quotient = math_ops.div(initial, denom)
        gamma = math_ops.cast(gamma, dtype)
        return math_ops.pow(quotient, gamma, name=name)


class Config(object):
    _stack = []

    def __str__(self):
        attrs = {k: getattr(self, k) for k in dir(self) if not k.startswith('_')}
        s = "<{} -\n{}\n>".format(self.__class__.__name__, pformat(attrs))
        return s

    def as_default(self):
        return context(self.__class__, self)


@contextmanager
def context(cls, obj):
    cls._stack.append(obj)
    yield
    cls._stack.pop()


def default_config():
    if not Config._stack:
        raise ValueError("Trying to get default config, but config stack is empty.")
    return Config._stack[-1]
