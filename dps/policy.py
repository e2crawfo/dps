from pprint import pformat

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.util.nest import flatten_dict_items


def placeholder_like(state_size, dtype=tf.float32, name=''):
    if isinstance(state_size, int):
        return tf.placeholder(dtype, (None, state_size), name=name)
    elif isinstance(state_size, tf.TensorShape):
        return tf.placeholder(dtype, (None,) + tuple(state_size), name=name)
    else:
        ph = [placeholder_like(ss, dtype, name="{}/{}".format(name, i)) for i, ss in enumerate(state_size)]
        return type(state_size)(*ph)


class Policy(RNNCell):
    """ Each instance of this class can be thought of as owning a set of variables. """
    existing_policies = set()

    def __init__(
            self, controller, action_selection, exploration,
            n_actions, obs_dim, name="policy"):

        self.controller = controller
        self.action_selection = action_selection
        self.exploration = exploration

        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.name = name

        self.n_builds = 0

        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope()

            self._policy_state = placeholder_like(self.controller.state_size, name='policy_state')
            self._obs = tf.placeholder(tf.float32, (None, self.obs_dim), name='obs')

        result = self.build(self._obs, self._policy_state)
        self._samples, self._action_activations, self._utils, self._next_policy_state = result

    def __str__(self):
        return pformat(self)

    def __call__(self, obs, policy_state):
        _, action_activations, utils, next_policy_state = self.build(obs, policy_state)
        return action_activations, next_policy_state

    def build(self, obs, policy_state):
        """ Return a Tensor giving the selected actions. """

        with tf.variable_scope(self.scope, reuse=self.n_builds > 0):
            utils, next_policy_state = self.controller(obs, policy_state)
            action_activations = self.action_selection(utils, self.exploration)

            if self.action_selection.can_sample:
                samples = sample_action(action_activations)
            else:
                samples = None

        self.n_builds += 1

        return samples, action_activations, utils, next_policy_state

    def build_feeddict(self, obs, policy_state):
        fd = flatten_dict_items({self._policy_state: policy_state})
        fd.update({self._obs: obs})
        return fd

    def act(self, obs, policy_state, sample=False):
        """ Samples actions. """
        sess = tf.get_default_session()
        feed_dict = self.build_feeddict(obs, policy_state)

        if sample:
            if not self.action_selection.can_sample:
                raise Exception("Attempting to sample from a policy that cannot be sampled from.")
            actions, next_policy_state = sess.run(
                [self._samples, self._next_policy_state], feed_dict)

        else:
            actions, next_policy_state = sess.run(
                [self._action_activations, self._next_policy_state], feed_dict)
        return actions, next_policy_state

    def zero_state(self, batch_size, dtype):
        return self.controller.zero_state(batch_size, dtype)

    @property
    def state_size(self):
        return self.controller.state_size

    @property
    def output_size(self):
        return self.n_actions


class ActionSelection(object):
    """ A callable mapping from utilities and an exploration param to action activation values.

    Attributes
    ----------
    can_sample: bool
        If True, returned action activations always represent a multinomial probability dist.

    """
    _can_sample = None

    @property
    def can_sample(self):
        assert isinstance(self._can_sample, bool), (
            "Subclasses of ActionSelection need to give boolean value to ``can_sample`` attribute.")
        return self._can_sample

    def __call__(self, utils, exploration):
        raise NotImplementedError()


class ReluSelect(ActionSelection):
    _can_sample = False

    def __call__(self, utils, temperature):
        return tf.nn.relu(utils, name="relu_action_selection")


class SoftmaxSelect(ActionSelection):
    _can_sample = True

    def __call__(self, utils, temperature):
        return tf.nn.softmax(utils/temperature, name="softmax_action_selection")


class GumbelSoftmaxSelect(ActionSelection):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Adapted from code by Eric Jang.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes

    """
    _can_sample = True

    def __init__(self, hard=False):
        self.hard = hard

    @staticmethod
    def _sample_gumbel(shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    @staticmethod
    def _gumbel_softmax_sample(logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + GumbelSoftmaxSelect._sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature, name='gumbel_softmax')

    def __call__(self, utils, temperature):
        y = self._gumbel_softmax_sample(utils, temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y


class EpsilonGreedySelect(ActionSelection):
    _can_sample = True

    def __call__(q_values, epsilon):
        mx = tf.reduce_max(q_values, axis=1, keep_dims=True)
        bool_is_max = tf.eq(q_values, mx)
        float_is_max = tf.cast(bool_is_max, tf.float32)
        max_count = tf.reduce_sum(float_is_max, axis=1, keep_dims=True)
        _probs = (float_is_max / max_count) * (1 - epsilon)
        n_actions = tf.shape(q_values)[1]
        probs = _probs + epsilon / tf.cast(n_actions, tf.float32)
        return probs


def sample_action(probs):
    logprobs = tf.log(probs)
    n_actions = tf.shape(logprobs)[1]
    samples = tf.cast(tf.multinomial(logprobs, 1), tf.int32)
    actions = tf.equal(samples, tf.reshape(tf.range(n_actions), (1, n_actions)))
    return tf.cast(actions, tf.float32)
