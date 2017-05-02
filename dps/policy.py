from pprint import pformat

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.util.nest import flatten_dict_items


def rnn_cell_placeholder(state_size, batch_size=None, dtype=tf.float32, name=''):
    if isinstance(state_size, int):
        return tf.placeholder(dtype, (batch_size, state_size), name=name)
    elif isinstance(state_size, tf.TensorShape):
        return tf.placeholder(dtype, (batch_size,) + tuple(state_size), name=name)
    else:
        ph = [
            rnn_cell_placeholder(
                ss, batch_size=batch_size, dtype=dtype, name="{}/{}".format(name, i))
            for i, ss in enumerate(state_size)]
        return type(state_size)(*ph)


class Policy(RNNCell):
    """ A map from observation-internal state pairs to action activations.

    Each instance of this class can be thought of as owning a set of variables,
    because whenever ``build`` is called, it will try to use the same set of variables
    (i.e. use the same scope, with ``reuse=True``) as the first time it was called.

    Parameters
    ----------
    controller: callable
        Outputs Tensor giving utilities.
    action_selection: callable
        Outputs Tensor giving action activations.
    exploration: Tensor
        Amount of exploration to use.
    n_actions: int
        Number of actions.
    obs_dim: int
        Dimension of observations.
    name: string
        Name of policy, used as name of variable scope.

    """
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

            self._policy_state = rnn_cell_placeholder(self.controller.state_size, name='policy_state')
            self._obs = tf.placeholder(tf.float32, (None, self.obs_dim), name='obs')

        result = self.build(self._obs, self._policy_state)
        self._samples, self._action_activations, self._utils, self._next_policy_state = result

    def __str__(self):
        return pformat(self)

    def __call__(self, obs, policy_state, sample=False):
        samples, action_activations, utils, next_policy_state = self.build(obs, policy_state)
        if sample and not self.action_selection.can_sample:
            raise ValueError(
                "Requesting samples from policy, but cannot sample from "
                "its action selection method: {}.".format(self.action_selection))
        if sample:
            return samples, next_policy_state
        else:
            return action_activations, next_policy_state

    def build(self, obs, policy_state, scope=None, reuse=True):
        reuse &= self.n_builds > 0
        scope = scope or self.scope
        with tf.variable_scope(scope, reuse=reuse):
            utils, next_policy_state = self.controller(obs, policy_state)
            action_activations = self.action_selection(utils, self.exploration)

            if self.action_selection.can_sample:
                samples = self.action_selection.sample(action_activations)
            else:
                samples = None

        self.n_builds += 1

        return samples, action_activations, utils, next_policy_state

    def build_feeddict(self, obs, policy_state):
        fd = flatten_dict_items({self._policy_state: policy_state})
        fd.update({self._obs: obs})
        return fd

    def act(self, obs, policy_state, sample=False):
        """ Return action activations given an observation and the current policy state.

        Perform arithmetical step of sampling from activation activations
        if ``sample`` is True and the activations represent a distribution
        over actions (so ``self.action_selection.can_sample`` is True).

        """
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

    def copy(self, new_name):
        new = Policy(
            self.controller, self.action_selection, self.exploration,
            self.n_actions, self.obs_dim, new_name)
        return new


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


class IdentitySelect(ActionSelection):
    _can_sample = False  # TODO this depends on what it gets as input

    def __call__(self, inp, epsilon):
        return tf.identity(inp, name="IdentitySelect")


class ReluSelect(ActionSelection):
    _can_sample = False

    def __call__(self, utils, temperature):
        return tf.nn.relu(utils, name="relu_action_selection")


class MultinomialSelect(ActionSelection):
    _can_sample = True

    def sample(self, action_activations):
        logprobs = tf.log(action_activations)
        n_actions = int(logprobs.shape[1])
        samples = tf.cast(tf.multinomial(logprobs, 1), tf.int32)
        reference = tf.reshape(tf.range(n_actions), (1, n_actions))
        actions = tf.equal(samples, reference)
        return tf.cast(actions, tf.float32)


class SoftmaxSelect(MultinomialSelect):
    def __call__(self, utils, temperature):
        softmax = tf.nn.softmax(utils/temperature, name="softmax_action_selection")
        return softmax


class GumbelSoftmaxSelect(MultinomialSelect):
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
        probs = tf.nn.softmax(y / temperature, name='gumbel_softmax')
        # probs = tf.Print(probs, [logits, y, probs], 'logits/noisy_logits/probs', summarize=7)
        return probs

    def __call__(self, utils, temperature):
        y = self._gumbel_softmax_sample(utils, temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
            # y_hard = tf.Print(y_hard, [y_hard], 'y_hard', summarize=7)
            y = tf.stop_gradient(y_hard - y) + y
        return y


class EpsilonGreedySelect(ActionSelection):
    def __call__(self, q_values, epsilon):
        mx = tf.reduce_max(q_values, axis=1, keep_dims=True)
        bool_is_max = tf.equal(q_values, mx)
        float_is_max = tf.cast(bool_is_max, tf.float32)
        max_count = tf.reduce_sum(float_is_max, axis=1, keep_dims=True)
        _probs = (float_is_max / max_count) * (1 - epsilon)
        n_actions = tf.shape(q_values)[1]
        probs = _probs + epsilon / tf.cast(n_actions, tf.float32)
        return probs


class BernoulliSelect(ActionSelection):
    _can_sample = True

    def sample(self, action_activations):
        uniform = tf.random_uniform(tf.shape(action_activations))
        return tf.cast(tf.less(uniform, action_activations), tf.float32)


class SigmoidSelect(BernoulliSelect):
    def __call__(self, utils, temperature):
        return tf.sigmoid(utils, name="sigmoid_action_selection")
