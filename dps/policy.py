from pprint import pformat
from copy import deepcopy

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
    """ A map from (observation, internal_state) pairs to action activations.

    Each instance of this class can be thought of as owning a set of variables,
    because whenever ``build`` is called, it will try to use the same set of variables
    (i.e. use the same scope, with ``reuse=True``) as the first time it was called.

    Note: in TensorFlow 1.1 (and probably 1.2), its fine to deepcopy an RNNCell up
    until the first time that it is used to add ops to the graph at which point
    variables are created (which are not ommitted by RNNCell.__deepcopy__ in TF 1.2).

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

        self.scope = None
        self.n_builds = 0
        self.act_is_built = False

    def __str__(self):
        return pformat(self)

    def __call__(self, obs, policy_state):
        samples, _, next_policy_state = self.build(obs, policy_state)
        return samples, next_policy_state

    def set_scope(self, scope):
        if self.n_builds > 0:
            raise ValueError("Cannot set scope once Policy has been built inside a graph.")
        self.scope = scope

    def capture_scope(self):
        """ Creates a variable scope called self.name inside current active scope, and stores it. """
        with tf.variable_scope(self.name):
            self.set_scope(tf.get_variable_scope())

    def get_scope(self):
        if not self.scope:
            self.capture_scope()
        return self.scope

    def build(self, obs, policy_state):
        with tf.variable_scope(self.get_scope(), reuse=self.n_builds > 0):
            utils, next_policy_state = self.controller(obs, policy_state)
            samples = self.action_selection.sample(utils, self.exploration)

        self.n_builds += 1

        return samples, utils, next_policy_state

    def build_sample(self, obs, policy_state):
        return self.build(obs, policy_state)

    def build_log_pdf(self, obs, policy_state, actions):
        with tf.variable_scope(self.get_scope(), reuse=self.n_builds > 0):
            utils, next_policy_state = self.controller(obs, policy_state)
            log_action_probabilities = self.action_selection.log_pdf(utils, actions, self.exploration)

        self.n_builds += 1

        return log_action_probabilities, next_policy_state

    def zero_state(self, batch_size, dtype):
        return self.controller.zero_state(batch_size, dtype)

    @property
    def state_size(self):
        return self.controller.state_size

    @property
    def output_size(self):
        return self.n_actions

    def deepcopy(self, new_name):
        if self.n_builds > 0:
            raise ValueError("Cannot copy Policy once it has been built inside a graph.")

        new = Policy(
            deepcopy(self.controller),
            deepcopy(self.action_selection),
            self.exploration,
            self.n_actions, self.obs_dim, new_name)
        return new
        # Should work in TF 1.1, will need to be changed in TF 1.2.
        # new_controller = deepcopy(self.controller)
        # if hasattr(new_controller, '_scope'):
        #     del new_controller._scope

        # new_as = deepcopy(self.action_selection)
        # if hasattr(new_as, '_scope'):
        #     del new_as._scope

        # new = Policy(
        #     deepcopy(self.controller),
        #     deepcopy(self.action_selection),
        #     self.exploration,
        #     self.n_actions, self.obs_dim, new_name)

        # if hasattr(new, '_scope'):
        #     del new._scope

        # return new

    def maybe_build_act(self):
        if not self.act_is_built:
            # Build a subgraph that we carry around with the Policy for implementing the ``act`` method
            self._policy_state = rnn_cell_placeholder(self.controller.state_size, name='policy_state')
            self._obs = tf.placeholder(tf.float32, (None, self.obs_dim), name='obs')

            result = self.build(self._obs, self._policy_state)
            self._samples, self._utils, self._next_policy_state = result

            self.act_is_built = True

    def build_feeddict(self, obs, policy_state):
        fd = flatten_dict_items({self._policy_state: policy_state})
        fd.update({self._obs: obs})
        return fd

    def act(self, obs, policy_state):
        """ Return action activations given an observation and the current policy state.

        """
        self.maybe_build_act()

        sess = tf.get_default_session()
        feed_dict = self.build_feeddict(obs, policy_state)

        actions, next_policy_state = sess.run([self._samples, self._next_policy_state], feed_dict)

        return actions, next_policy_state


class ActionSelection(object):
    """ A callable mapping from utilities and an exploration param to action activation values.

    """
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.n_params = n_actions

    def sample(self, utils, exploration):
        raise Exception()

    def log_pdf(self, utils, actions, exploration):
        raise Exception()

    def kl_divergence(self, utils1, utils2, exploration):
        raise Exception()


class ProductSelection(object):
    def __init__(self, components):
        self.components = components
        self.n_params_vector = [c.n_params for c in components]
        self.n_params = sum(self.n_params_vector)
        self.n_actions_vector = [c.n_actions for c in components]
        self.n_actions = sum(self.n_actions_vector)

    def sample(self, utils, exploration):
        _utils = tf.split(utils, self.n_params_vector, axis=-1)
        return tf.concat(
            [c.sample(u, exploration) for u, c in zip(_utils, self.components)],
            axis=-1)

    def log_pdf(self, utils, actions, exploration):
        _utils = tf.split(utils, self.n_params_vector, axis=-1)
        _actions = tf.split(actions, self.n_actions_vector, axis=-1)
        _log_pdf = tf.concat(
            [c.log_pdf(u, a, exploration) for u, a, c in zip(_utils, _actions, self.components)],
            axis=-1)
        return tf.reduce_sum(_log_pdf, axis=-1, keep_dims=True)

    def kl_divergence(self, utils1, utils2, exploration):
        _utils1 = tf.split(utils1, self.n_params_vector, axis=-1)
        _utils2 = tf.split(utils2, self.n_params_vector, axis=-1)

        _splitwise_kl = tf.concate(
            [c.kl_divergence(u1, u2, exploration) for u1, u2, c in zip(_utils1, _utils2, self.components)],
            axis=-1)

        return tf.reduce_sum(_splitwise_kl, axis=-1, keep_dims=True)


class Normal(ActionSelection):
    def __init__(self):
        self.n_actions = 1
        self.n_params = 2

    def _dist(self, utils, exploration):
        mean = utils[:, 0:1]
        # std = exploration * tf.exp(utils[:, 1:2])
        std = tf.exp(utils[:, 1:2])
        dist = tf.contrib.distributions.Normal(loc=mean, scale=std)
        return dist

    def sample(self, utils, exploration=1.0):
        dist = self._dist(utils, exploration)
        return dist.sample()

    def log_pdf(self, utils, actions, exploration=1.0):
        dist = self._dist(utils, exploration)
        return dist.log_prob(actions)

    def kl_divergence(self, utils1, utils2, exploration):
        mean1, std1 = utils1[:, 0:1], tf.exp(utils1[:, 1:2])
        mean2, std2 = utils2[:, 0:1], tf.exp(utils2[:, 1:2])
        return tf.log(std2/std1) + (std1**2 + (mean1 - mean2)**2) / 2*std2**2 - 0.5


class Gamma(ActionSelection):
    """ alpha, beta """
    def __init__(self):
        self.n_actions = 1
        self.n_params = 2

    def _dist(self, utils, exploration):
        alpha = tf.exp(utils[:, 0:1])
        beta = exploration * tf.exp(utils[:, 1:2])
        dist = tf.contrib.distributions.Gamma(concentration=alpha, rate=beta)
        return dist

    def sample(self, utils, exploration=1.0):
        dist = self._dist(utils, exploration)
        return dist.sample()

    def log_pdf(self, utils, actions, exploration=1.0):
        dist = self._dist(utils, exploration)
        return dist.log_prob(actions)

    def kl_divergence(self, utils1, utils2, exploration):
        alpha1, beta1 = utils1[:, 0:1], exploration * tf.exp(utils1[:, 1:2])
        alpha2, beta2 = utils2[:, 0:1], exploration * tf.exp(utils2[:, 1:2])
        return (
            (alpha1 - alpha2) * tf.digamma(alpha1) -
            tf.lgamma(alpha1) + tf.lgamma(alpha2) +
            alpha2 * (tf.log(beta1) - tf.log(beta2)) +
            alpha1 * (beta2 - beta1) / beta1)


class IdentitySelect(ActionSelection):
    def sample(self, inp, epsilon):
        return tf.identity(inp, name="IdentitySelect")


class ReluSelect(ActionSelection):
    def sample(self, utils, temperature):
        return tf.nn.relu(utils, name="relu_action_selection")


class SoftmaxSelect(ActionSelection):
    def sample(self, utils, temperature):
        logits = utils / temperature
        samples = tf.cast(tf.multinomial(logits, 1), tf.int32)
        actions = tf.one_hot(tf.reshape(samples, (-1,)), self.n_actions)
        return actions

    def log_pdf(self, utils, actions, temperature):
        """ Assumes each row of `actions` is a one-hot vector. """
        logits = utils / temperature
        return (tf.reduce_sum(logits * actions, axis=-1, keep_dims=True) -
                tf.reduce_logsumexp(logits, axis=-1, keep_dims=True))

    def kl_divergence(self, utils1, utils2, temperature):
        logits1 = utils1 / temperature
        logits2 = utils2 / temperature

        log_norm1 = tf.reduce_logsumexp(logits1, axis=-1, keep_dims=True)
        norm1 = tf.exp(log_norm1)
        log_norm2 = tf.reduce_logsumexp(logits2, axis=-1, keep_dims=True)

        return -tf.reduce_sum(tf.exp(logits1) * (logits1 - logits2)) / norm1 - log_norm2 + log_norm1


class EpsilonGreedySelect(ActionSelection):
    def _probs(self, utils, epsilon):
        q_values = utils
        mx = tf.reduce_max(q_values, axis=1, keep_dims=True)
        bool_is_max = tf.equal(q_values, mx)
        float_is_max = tf.cast(bool_is_max, tf.float32)
        max_count = tf.reduce_sum(float_is_max, axis=1, keep_dims=True)
        _probs = (float_is_max / max_count) * (1 - epsilon)
        n_actions = tf.shape(q_values)[1]
        probs = _probs + epsilon / tf.cast(n_actions, tf.float32)
        return probs

    def sample(self, utils, epsilon):
        probs = self._probs(utils, epsilon)
        logprobs = tf.log(probs)
        samples = tf.cast(tf.multinomial(logprobs, 1), tf.int32)
        actions = tf.one_hot(tf.reshape(samples, (-1,)), self.n_actions)
        return actions

    def log_pdf(self, utils, actions, epsilon):
        """ Assumes each row of `action` is a one-hot vector. """
        probs = self._probs(utils, epsilon)
        action_probs = tf.reduce_sum(actions * probs, axis=-1, keep_dims=True)
        return tf.log(action_probs)


class BernoulliSelect(ActionSelection):
    def sample(self, utils, exploration):
        probs = tf.sigmoid(utils, name="bernoulli_select")
        uniform = tf.random_uniform(tf.shape(probs))
        return tf.cast(tf.less(uniform, probs), tf.float32)


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
    def __init__(self, n_actions, hard=False):
        self.hard = hard
        super(GumbelSoftmaxSelect, self).__init__(n_actions)

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

    def sample(self, utils, temperature):
        y = self._gumbel_softmax_sample(utils, temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
