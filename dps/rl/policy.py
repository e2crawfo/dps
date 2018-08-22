import tensorflow as tf
from tensorflow.python.util.nest import flatten_dict_items
import tensorflow.contrib.distributions as tf_dists

from dps import cfg
from dps.rl import AgentHead
from dps.utils.tf import (
    MLP, FeedforwardCell, CompositeCell, tf_roll,
    masked_mean, build_scheduled_value
)


class _DoWeightingValue(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, reward_sum, inp):
        weight, reward = inp
        return weight * (reward + self.gamma * reward_sum)


class _DoWeightingActionValue(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, reward_sum, inp):
        weight, reward = inp
        return reward + weight * self.gamma * reward_sum


class Policy(AgentHead):
    def __init__(
            self, action_selection, obs_shape, exploration_schedule,
            val_exploration_schedule=None, name=None):

        self.action_selection = action_selection
        self.param_shape = self.action_selection.param_shape
        self.action_shape = self.action_selection.action_shape
        self.obs_shape = obs_shape

        self.act_is_built = False
        self.train_exploration_schedule = exploration_schedule
        self.val_exploration_schedule = val_exploration_schedule

        self.mode = None

        super(Policy, self).__init__(name)

    @property
    def n_params(self):
        assert len(self.action_selection.param_shape) == 1
        return self.action_selection.param_shape[0]

    def __call__(self, obs, controller_state):
        utils, next_controller_state = self.agent.get_one_step_utils(obs, controller_state, self.name)
        entropy = self.action_selection.entropy(utils, self.exploration)
        samples = self.action_selection.sample(utils, self.exploration)
        log_probs = self.action_selection.log_probs(utils, samples, self.exploration)
        return (log_probs, samples, entropy, utils), next_controller_state

    def maybe_build_mode(self):
        if self.mode is None:
            self.mode = tf.Variable("", dtype=tf.string, name="{}-mode".format(self.display_name))
            self._mode_input = tf.placeholder(tf.string, ())
            self._mode_setter = tf.assign(self.mode, self._mode_input)

    def set_mode(self, mode):
        self.maybe_build_mode()
        tf.get_default_session().run(self._mode_setter, feed_dict={self._mode_input: mode})

    def build_core_signals(self, context):
        self.maybe_build_mode()

        self.train_exploration = build_scheduled_value(self.train_exploration_schedule)
        if self.val_exploration_schedule is None:
            self.exploration = self.val_exploration = self.train_exploration
        else:
            self.val_exploration = build_scheduled_value(self.val_exploration_schedule)
            self.exploration = tf.cond(
                tf.logical_or(
                    tf.equal(self.mode, "train"),
                    tf.equal(self.mode, "off_policy"),
                ),
                lambda: self.train_exploration,
                lambda: self.val_exploration
            )
        label = "{}-exploration".format(self.display_name)
        context.add_recorded_value(label, self.exploration)

    def generate_signal(self, key, context, **kwargs):
        if key == 'log_probs':
            utils = self.agent.get_utils(self, context)
            actions = context.get_signal('actions')
            return self.action_selection.log_probs(utils, actions, self.exploration)
        elif key == 'entropy':
            utils = self.agent.get_utils(self, context)
            return self.action_selection.entropy(utils, self.exploration)
        elif key == 'samples':
            utils = self.agent.get_utils(self, context)
            return self.action_selection.sample(utils, self.exploration)
        elif key == 'kl':
            raise Exception("NotImplemented")
        elif key in ['monte_carlo_values', 'monte_carlo_action_values']:
            c = kwargs.get('c', None)
            rho = context.get_signal('rho', self, c=c)
            rewards = context.get_signal('rewards')

            if key == 'monte_carlo_action_values':
                rho = tf_roll(rho, 1, fill=1.0, reverse=True)

            gamma = context.get_signal('gamma')

            elems = (
                tf.reverse(rho, axis=[0]),
                tf.reverse(rewards, axis=[0])
            )

            initializer = tf.zeros_like(rewards[0, ...])

            if key == 'monte_carlo_action_values':
                func = _DoWeightingActionValue(gamma)
            else:
                func = _DoWeightingValue(gamma)

            returns = tf.scan(
                func,
                elems=elems,
                initializer=initializer,
            )

            returns = tf.reverse(returns, axis=[0])
            return returns
        elif key == 'average_monte_carlo_values':
            values = context.get_signal('monte_carlo_values', self, **kwargs)
            average = tf.reduce_mean(values, axis=1, keepdims=True)
            average += tf.zeros_like(values)
            return average
        elif key == 'importance_weights':
            pi_log_probs = context.get_signal("log_probs", self)
            mu_log_probs = context.get_signal("mu_log_probs")
            importance_weights = tf.exp(pi_log_probs - mu_log_probs)

            label = "{}-mean_importance_weight".format(self.display_name)
            mask = context.get_signal("mask")
            context.add_recorded_value(label, masked_mean(importance_weights, mask))

            return importance_weights
        elif key == 'rho':
            c = kwargs.get('c', None)
            importance_weights = context.get_signal("importance_weights", self)

            if c is not None:
                if c <= 0:
                    rho = importance_weights
                else:
                    rho = tf.minimum(importance_weights, c)
            else:
                rho = tf.ones_like(importance_weights)

            label = "{}-mean_rho_c_{}".format(self.display_name, c)
            mask = context.get_signal("mask")
            context.add_recorded_value(label, masked_mean(rho, mask))

            return rho
        else:
            raise Exception("NotImplemented")

    def maybe_build_act(self):
        if not self.act_is_built:
            # Build a subgraph that we carry around with the Policy for implementing the ``act`` method
            self._policy_state = rnn_cell_placeholder(self.agent.controller.state_size, name='policy_state')
            self._obs = tf.placeholder(tf.float32, (None,)+self.obs_shape, name='obs')
            (
                (self._log_probs, self._samples, self._entropy, self._utils),
                self._next_policy_state
            ) = self(self._obs, self._policy_state)

            self.act_is_built = True

    def act(self, obs, policy_state, exploration=None):
        """ Return (actions, next policy state) given an observation and the current policy state. """
        self.maybe_build_mode()
        self.maybe_build_act()

        sess = tf.get_default_session()
        feed_dict = flatten_dict_items({self._policy_state: policy_state})
        feed_dict.update({self._obs: obs})
        if exploration is not None:
            feed_dict.update({self.exploration: exploration})

        log_probs, actions, entropy, utils, next_policy_state = sess.run(
            [self._log_probs, self._samples, self._entropy,
             self._utils, self._next_policy_state],
            feed_dict=feed_dict)

        return (log_probs, actions, entropy, utils), next_policy_state

    @property
    def state_size(self):
        return self.agent.controller.state_size

    def zero_state(self, *args, **kwargs):
        return self.agent.controller.zero_state(*args, **kwargs)


class DiscretePolicy(Policy):
    def __init__(self, action_selection, *args, **kwargs):
        assert isinstance(action_selection, Categorical)
        super(DiscretePolicy, self).__init__(action_selection, *args, **kwargs)

    def generate_signal(self, key, context, **kwargs):
        if key == 'log_probs_all':
            utils = self.agent.get_utils(self, context)
            return self.action_selection.log_probs_all(utils, self.exploration)
        else:
            return super(DiscretePolicy, self).generate_signal(key, context, **kwargs)


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


class ActionSelection(object):
    def __init__(self, action_shape):
        self.action_shape = action_shape
        self.param_shape = action_shape

    def sample(self, utils, exploration):
        s = self._sample(utils, exploration)
        return tf.identity(s, name="{}.sample".format(self.__class__.__name__))

    def log_probs(self, utils, actions, exploration):
        lp = self._log_probs(utils, actions, exploration)
        return tf.identity(lp, name="{}.log_probs".format(self.__class__.__name__))

    def entropy(self, utils, exploration):
        e = self._entropy(utils, exploration)
        return tf.identity(e, name="{}.entropy".format(self.__class__.__name__))

    def kl(self, utils1, utils2, exploration):
        kl = self._kl(utils1, utils2, exploration)
        return tf.identity(kl, name="{}.kl".format(self.__class__.__name__))

    def _sample(self, utils, exploration):
        raise Exception()

    def _log_probs(self, utils, actions, exploration):
        raise Exception()

    def _entropy(self, utils, exploration):
        raise Exception()

    def _kl(self, utils1, utils2, exploration):
        raise Exception()


class ProductDist(ActionSelection):
    def __init__(self, *components):
        self.components = components

        for c in components:
            assert len(c.param_shape) == 1
            assert len(c.action_shape) == 1

        self.param_dim_vector = [c.param_shape[0] for c in components]
        self.param_dim = sum(self.param_dim_vector)
        self.action_dim_vector = [c.action_shape[0] for c in components]
        self.action_dim = sum(self.action_dim_vector)

        self.param_shape = (self.param_dim,)
        self.action_shape = (self.action_dim,)

    def _sample(self, utils, exploration):
        _utils = tf.split(utils, self.param_dim_vector, axis=-1)
        _samples = [tf.to_float(c.sample(u, exploration)) for u, c in zip(_utils, self.components)]
        return tf.concat(_samples, axis=1)

    def _log_probs(self, utils, actions, exploration):
        _utils = tf.split(utils, self.param_dim_vector, axis=-1)
        _actions = tf.split(actions, self.action_dim_vector, axis=-1)
        _log_probs = [c.log_probs(u, a, exploration) for u, a, c in zip(_utils, _actions, self.components)]
        return tf.reduce_sum(tf.concat(_log_probs, axis=-1), axis=-1, keepdims=True)

    def _entropy(self, utils, exploration):
        _utils = tf.split(utils, self.param_dim_vector, axis=-1)
        _entropies = [c.entropy(u, exploration) for u, c in zip(_utils, self.components)]
        return tf.reduce_sum(tf.concat(_entropies, axis=-1), axis=-1, keepdims=True)

    def _kl(self, utils1, utils2, e1, e2=None):
        _utils1 = tf.split(utils1, self.param_dim_vector, axis=-1)
        _utils2 = tf.split(utils2, self.param_dim_vector, axis=-1)

        _splitwise_kl = tf.concat(
            [c.kl(u1, u2, e1, e2)
             for u1, u2, c in zip(_utils1, _utils2, self.components)],
            axis=-1)

        return tf.reduce_sum(_splitwise_kl, axis=-1, keepdims=True)


class TensorFlowSelection(ActionSelection):
    def _dist(self, utils, exploration):
        raise Exception()

    def _sample(self, utils, exploration):
        dist = self._dist(utils, exploration)
        samples = tf.cast(dist.sample(), tf.float32)
        if len(dist.event_shape) == 0:
            samples = samples[..., None]
        return samples

    def _log_probs(self, utils, actions, exploration):
        dist = self._dist(utils, exploration)
        actions = tf.reshape(actions, tf.shape(dist.sample()))
        return dist.log_prob(actions)[..., None]

    def _entropy(self, utils, exploration):
        dist = self._dist(utils, exploration)
        return dist.entropy()[..., None]

    def _kl(self, utils1, utils2, e1, e2=None):
        e2 = e1 if e2 is None else e2
        dist1 = self._dist(utils1, e1)
        dist2 = self._dist(utils2, e2)
        return tf_dists.kl(dist1, dist2)[..., None]


class Deterministic(TensorFlowSelection):
    def __init__(self, param_shape, action_shape=None, func=None):
        self.param_shape = param_shape
        self.action_shape = action_shape or param_shape
        self.func = func or (lambda x: tf.identity(x))

    def _dist(self, utils, exploration):
        return tf_dists.VectorDeterministic(self.func(utils))

    def _entropy(self, utils, exploration):
        entropy = tf.zeros(tf.shape(utils)[:-1])[..., None]
        return tf.identity(entropy, name="{}.entropy".format(self.__class__.__name__))

    def _kl(self, utils1, utils2, e1, e2=None):
        raise Exception("NotImplemented")


def softplus(x):
    return tf.log(1 + tf.exp(x))


class SigmoidBeta(TensorFlowSelection):
    def __init__(self, c0_bounds, c1_bounds):
        self.action_shape = (1,)
        self.param_shape = (2,)
        self.c0_bounds = c0_bounds
        self.c1_bounds = c1_bounds

    def _dist(self, utils, exploration):
        c0 = tf.sigmoid(utils[..., 0]) * (self.c0_bounds[1] - self.c0_bounds[0]) + self.c0_bounds[0]
        c1 = tf.sigmoid(utils[..., 1]) * (self.c1_bounds[1] - self.c1_bounds[0]) + self.c1_bounds[0]
        return tf_dists.Beta(concentration0=c0, concentration1=c1)


class Normal(TensorFlowSelection):
    def __init__(self, explore=False):
        self.action_shape = (1,)
        self.param_shape = (1,) if explore else (2,)
        self.explore = explore

    def _dist(self, utils, exploration):
        mean = utils[..., 0]
        scale = exploration if self.explore else softplus(utils[..., 1])
        return tf_dists.Normal(loc=mean, scale=scale)


class SigmoidNormal(TensorFlowSelection):
    def __init__(self, low=0.0, high=1.0, explore=False):
        self.action_shape = (1,)
        self.param_shape = (1,) if explore else (2,)
        self.low = low
        self.high = high
        self.explore = explore

    def _dist(self, utils, exploration):
        mean = tf.nn.sigmoid(utils[..., 0]) * (self.high - self.low) + self.low
        scale = exploration if self.explore else softplus(utils[..., 1])
        return tf_dists.Normal(loc=mean, scale=scale)


class Gamma(TensorFlowSelection):
    """ alpha, beta """
    def __init__(self):
        self.action_shape = (1,)
        self.param_shape = (2,)

    def _dist(self, utils, exploration):
        concentration = softplus(utils[..., 0])
        rate = softplus(utils[..., 1])

        return tf_dists.Gamma(concentration=concentration, rate=rate)


class Beta(TensorFlowSelection):
    def __init__(self, offset=None, maximum=None):
        self.action_shape = (1,)
        self.param_shape = (2,)
        self.offset = offset
        self.maximum = maximum

    def _dist(self, utils, exploration):
        c0 = softplus(utils[..., 0])
        c1 = softplus(utils[..., 1])

        if self.offset is not None:
            c0 += self.offset
            c1 += self.offset

        if self.maximum is not None:
            c0 += tf.minimum(c0, self.maximum)
            c1 += tf.minimum(c1, self.maximum)

        return tf_dists.Beta(concentration0=c0, concentration1=c1)


class BetaMeanESS(TensorFlowSelection):
    """ A Beta distribution parameterized by mean and effective sample size. """

    def __init__(self):
        self.action_shape = (1,)
        self.param_shape = (2,)

    def _dist(self, utils, exploration):
        mean = tf.nn.sigmoid(utils[..., 0])
        ess = softplus(utils[..., 1])

        c1 = mean * ess
        c0 = (1 - mean) * ess

        return tf_dists.Beta(concentration0=c0, concentration1=c1)


class Bernoulli(TensorFlowSelection):
    def __init__(self):
        self.action_shape = self.param_shape = (1,)

    def _dist(self, utils, exploration):
        return tf_dists.BernoulliWithSigmoidProbs(utils[..., 0])


class Categorical(TensorFlowSelection):
    def __init__(self, n_actions, one_hot=True):
        self.param_shape = (n_actions,)
        self.action_shape = (n_actions,) if one_hot else (1,)
        self.n_actions = n_actions

        self.one_hot = one_hot

    def _sample(self, utils, exploration):
        dist = self._dist(utils, exploration)
        sample = dist.sample()
        if self.one_hot:
            sample = tf.one_hot(sample, depth=self.n_actions, axis=-1)
        else:
            sample = sample[..., None]
        return sample

    def _log_probs(self, utils, actions, exploration):
        dist = self._dist(utils, exploration)
        if self.one_hot:
            actions = tf.argmax(actions, axis=-1)
        else:
            sample = dist.sample()
            actions = tf.reshape(actions, tf.shape(sample))
            actions = tf.cast(actions, sample.dtype)
        return dist.log_prob(actions)[..., None]

    def log_probs_all(self, utils, exploration):
        dist = self._dist(utils, exploration)

        batch_rank = len(utils.shape)-1
        sample_shape = self.action_shape + (1,) * batch_rank
        sample = tf.reshape(tf.range(self.action_shape), sample_shape)
        log_probs = dist.log_prob(sample)
        axis_perm = tuple(range(1, batch_rank+1)) + (0,)
        log_probs = tf.transpose(log_probs, perm=axis_perm)
        return tf.identity(log_probs, name="{}.log_probs_all".format(self.__class__.__name__))

    def _entropy(self, utils, exploration):
        dist = self._dist(utils, exploration)
        return dist.entropy()[..., None]

    def _kl(self, utils1, utils2, e1, e2=None):
        e2 = e1 if e2 is None else e2
        dist1 = self._dist(utils1, e1)
        dist2 = self._dist(utils2, e2)
        return tf_dists.kl(dist1, dist2)[..., None]

    def _dist(self, utils, exploration):
        raise Exception("NotImplemented")

    @staticmethod
    def _build_categorical_dist(*args, probs=None, **kwargs):
        """ Build a non-one-hot categorical distribution.

        Do this to avoid using tf_dists.OneHotCategorical even if `one_hot` is True,
        because it doesn't work in tf versions <= 1.0.0.

        """
        if tf.__version__ < "1.1":
            return tf_dists.Categorical(*args, p=probs, **kwargs)
        else:
            return tf_dists.Categorical(*args, probs=probs, **kwargs)


class FixedCategorical(Categorical):
    def __init__(self, action_shape, probs=None, logits=None, one_hot=True):
        assert (probs is None) != (logits is None)
        self.param_shape = 0
        self.action_shape = action_shape if one_hot else 1
        self.probs = probs
        self.logits = logits
        self.one_hot = one_hot

    def _dist(self, utils, exploration):
        return self._build_categorical_dist(logits=utils/exploration)


class Softmax(Categorical):
    def _dist(self, utils, exploration):
        return self._build_categorical_dist(logits=utils/exploration)


class EpsilonSoftmax(Categorical):
    """ Mixture between a softmax distribution and a uniform distribution.
        Weight of the uniform distribution is given by the exploration
        coefficient epsilon, and the softmax uses a temperature of 1.

    """
    def __init__(self, *args, softmax_temp=1.0, **kwargs):
        self.softmax_temp = softmax_temp
        super(EpsilonSoftmax, self).__init__(*args, **kwargs)

    def _dist(self, utils, epsilon):
        probs = (1 - epsilon) * tf.nn.softmax(utils/self.softmax_temp) + epsilon / self.n_actions
        return self._build_categorical_dist(probs=probs)


class EpsilonGreedy(EpsilonSoftmax):
    def __init__(self, *args, **kwargs):
        super(EpsilonGreedy, self).__init__(*args, softmax_temp=0.1, **kwargs)


class BuildLstmController(object):
    def __call__(self, param_shape, name=None):
        return CompositeCell(
            tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
            MLP(), param_shape, name=name)


class BuildMlpController(object):
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, param_shape, name=None):
        return FeedforwardCell(MLP(*self.args, **self.kwargs), param_shape, name=name)


class BuildLinearController(object):
    def __init__(self):
        pass

    def __call__(self, param_shape, name=None):
        return FeedforwardCell(MLP(), param_shape, name=name)


class _BuildDiscretePolicy(object):
    def __init__(self, one_hot=True):
        self.one_hot = one_hot

    def __call__(self, env, **kwargs):
        if self.one_hot:
            n_actions = env.action_shape[0]
        else:
            n_actions = env.n_actions
            assert env.action_shape == (1,)
        action_selection = self.action_selection_klass(n_actions, one_hot=self.one_hot)
        return DiscretePolicy(action_selection, env.obs_shape, **kwargs)


class BuildSoftmaxPolicy(_BuildDiscretePolicy):
    action_selection_klass = Softmax


class BuildEpsilonSoftmaxPolicy(_BuildDiscretePolicy):
    action_selection_klass = EpsilonSoftmax


class BuildEpsilonGreedyPolicy(_BuildDiscretePolicy):
    action_selection_klass = EpsilonGreedy
