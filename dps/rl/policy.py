import tensorflow as tf
from tensorflow.python.util.nest import flatten_dict_items
import tensorflow.contrib.distributions as tf_dists

from dps import cfg
from dps.rl import AgentHead
from dps.utils.tf import (
    MLP, FeedforwardCell, CompositeCell, tf_roll,
    masked_mean, build_scheduled_value
)


class BuildLstmController(object):
    def __call__(self, params_dim, name=None):
        return CompositeCell(
            tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
            MLP(), params_dim, name=name)


class BuildFeedforwardController(object):
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, params_dim, name=None):
        return FeedforwardCell(MLP(*self.args, **self.kwargs), params_dim, name=name)


class BuildLinearController(object):
    def __init__(self):
        pass

    def __call__(self, params_dim, name=None):
        return FeedforwardCell(MLP(), params_dim, name=name)


class BuildSoftmaxPolicy(object):
    def __init__(self, one_hot=True):
        self.one_hot = one_hot

    def __call__(self, env, **kwargs):
        n_actions = env.actions_dim if self.one_hot else env.n_actions
        if not self.one_hot:
            assert env.actions_dim == 1
        return DiscretePolicy(Softmax(n_actions, one_hot=self.one_hot), env.obs_shape, **kwargs)


class BuildEpsilonGreedyPolicy(object):
    def __init__(self, one_hot=True):
        self.one_hot = one_hot

    def __call__(self, env, **kwargs):
        n_actions = env.actions_dim if self.one_hot else env.n_actions
        if not self.one_hot:
            assert env.actions_dim == 1
        return DiscretePolicy(EpsilonGreedy(n_actions, one_hot=self.one_hot), env.obs_shape, **kwargs)


class BuildEpsilonSoftmaxPolicy(object):
    def __init__(self, one_hot=True):
        self.one_hot = one_hot

    def __call__(self, env, **kwargs):
        n_actions = env.actions_dim if self.one_hot else env.n_actions
        if not self.one_hot:
            assert env.actions_dim == 1
        return DiscretePolicy(EpsilonSoftmax(n_actions, one_hot=self.one_hot), env.obs_shape, **kwargs)


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
            self, action_selection, obs_shape,
            exploration_schedule,
            val_exploration_schedule=None, name=None):

        self.action_selection = action_selection
        self.params_dim = self.action_selection.params_dim
        self.actions_dim = self.action_selection.actions_dim
        self.obs_shape = obs_shape

        self.act_is_built = False
        self.train_exploration_schedule = exploration_schedule
        self.val_exploration_schedule = val_exploration_schedule

        self.mode = None

        super(Policy, self).__init__(name)

    @property
    def size(self):
        return self.action_selection.params_dim

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
                    tf.equal(self.mode, "update"),
                ),
                lambda: self.train_exploration,
                lambda: self.val_exploration
            )
        label = "{}-exploration".format(self.display_name)
        context.add_summary(tf.summary.scalar(label, self.exploration, collections=['scheduled_value_summaries']))

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
            if context.truncated_rollouts:
                raise Exception("NotImplemented")

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
            average = tf.reduce_mean(values, axis=1, keep_dims=True)
            average += tf.zeros_like(values)
            return average
        elif key == 'importance_weights':
            pi_log_probs = context.get_signal("log_probs", self)
            mu_log_probs = context.get_signal("mu_log_probs")
            importance_weights = tf.exp(pi_log_probs - mu_log_probs)

            label = "{}-mean_importance_weight".format(self.display_name)
            mask = context.get_signal("mask")
            context.add_summary(tf.summary.scalar(label, masked_mean(importance_weights, mask)))

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

            label = "{}-mean_rho_c={}".format(self.display_name, c)
            mask = context.get_signal("mask")
            context.add_summary(tf.summary.scalar(label, masked_mean(rho, mask)))

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
    def __init__(self, actions_dim):
        self.actions_dim = actions_dim
        self.params_dim = actions_dim

    def sample(self, utils, exploration):
        raise Exception()

    def log_probs(self, utils, actions, exploration):
        raise Exception()

    def kl(self, utils1, utils2, exploration):
        raise Exception()


def final_d(tensor):
    """ tf.split does not accept axis=-1, so use this instead. """
    return len(tensor.shape)-1


class ProductDist(ActionSelection):
    def __init__(self, *components):
        self.components = components
        self.params_dim_vector = [c.params_dim for c in components]
        self.params_dim = sum(self.params_dim_vector)
        self.actions_dim_vector = [c.actions_dim for c in components]
        self.actions_dim = sum(self.actions_dim_vector)

    def sample(self, utils, exploration):
        _utils = tf.split(utils, self.params_dim_vector, axis=final_d(utils))
        _samples = [c.sample(u, exploration) for u, c in zip(_utils, self.components)]
        return tf.concat(_samples, axis=1)

    def log_probs(self, utils, actions, exploration):
        _utils = tf.split(utils, self.params_dim_vector, axis=final_d(utils))
        _actions = tf.split(actions, self.actions_dim_vector, axis=final_d(actions))
        _log_probs = [c.log_probs(u, a, exploration) for u, a, c in zip(_utils, _actions, self.components)]
        return tf.reduce_sum(tf.concat(_log_probs, axis=-1), axis=-1, keep_dims=True)

    def entropy(self, utils, exploration):
        _utils = tf.split(utils, self.params_dim_vector, axis=final_d(utils))
        _entropies = [c.entropy(u, exploration) for u, c in zip(_utils, self.components)]
        return tf.reduce_sum(tf.concat(_entropies, axis=-1), axis=-1, keep_dims=True)

    def kl(self, utils1, utils2, e1, e2=None):
        _utils1 = tf.split(utils1, self.params_dim_vector, axis=final_d(utils1))
        _utils2 = tf.split(utils2, self.params_dim_vector, axis=final_d(utils2))

        _splitwise_kl = tf.concat(
            [c.kl(u1, u2, e1, e2)
             for u1, u2, c in zip(_utils1, _utils2, self.components)],
            axis=-1)

        return tf.reduce_sum(_splitwise_kl, axis=-1, keep_dims=True)


class TensorFlowSelection(ActionSelection):
    def _dist(self, utils, exploration):
        raise Exception()

    def sample(self, utils, exploration):
        dist = self._dist(utils, exploration)
        samples = tf.cast(dist.sample(), tf.float32)
        if len(dist.event_shape) == 0:
            samples = samples[..., None]
        return samples

    def log_probs(self, utils, actions, exploration):
        dist = self._dist(utils, exploration)
        actions = tf.reshape(actions, tf.shape(dist.sample()))
        return dist.log_prob(actions)[..., None]

    def entropy(self, utils, exploration):
        dist = self._dist(utils, exploration)
        return dist.entropy()[..., None]

    def kl(self, utils1, utils2, e1, e2=None):
        e2 = e1 if e2 is None else e2
        dist1 = self._dist(utils1, e1)
        dist2 = self._dist(utils2, e2)
        return tf_dists.kl(dist1, dist2)[..., None]


class Deterministic(TensorFlowSelection):
    def __init__(self, params_dim, actions_dim=None, func=None):
        self.params_dim = params_dim
        self.actions_dim = actions_dim or params_dim
        self.func = func or (lambda x: tf.identity(x))

    def _dist(self, utils, exploration):
        return tf_dists.VectorDeterministic(self.func(utils))

    def entropy(self, utils, exploration):
        return tf.zeros(tf.shape(utils)[:-1])[..., None]

    def kl(self, utils1, utils2, e1, e2=None):
        raise Exception("NotImplemented")


def softplus(x):
    return tf.log(1 + tf.exp(x))


class Normal(TensorFlowSelection):
    def __init__(self):
        self.actions_dim = 1
        self.params_dim = 2

    def _dist(self, utils, exploration):
        mean = utils[..., 0]
        scale = softplus(utils[..., 1])
        # Could use tf_dists.NormalWithSoftplusScale, but found it to cause problems
        # when taking hessian-vector products.
        dist = tf_dists.Normal(loc=mean, scale=scale)
        return dist


class NormalWithFixedScale(TensorFlowSelection):
    def __init__(self, scale):
        self.actions_dim = 1
        self.params_dim = 1
        self.scale = scale

    def _dist(self, utils, exploration):
        dist = tf_dists.Normal(loc=utils[..., 0], scale=self.scale)
        return dist


class NormalWithExploration(TensorFlowSelection):
    def __init__(self):
        self.actions_dim = 1
        self.params_dim = 1

    def _dist(self, utils, exploration):
        dist = tf_dists.Normal(loc=utils[..., 0], scale=exploration)
        return dist


class Gamma(TensorFlowSelection):
    """ alpha, beta """
    def __init__(self):
        self.actions_dim = 1
        self.params_dim = 2

    def _dist(self, utils, exploration):
        concentration = softplus(utils[..., 0])
        rate = softplus(utils[..., 1])

        dist = tf_dists.Gamma(concentration=concentration, rate=rate)
        return dist


class Bernoulli(TensorFlowSelection):
    def __init__(self):
        self.actions_dim = self.params_dim = 1

    def _dist(self, utils, exploration):
        return tf_dists.BernoulliWithSigmoidProbs(utils[..., 0])


def build_categorical_dist(*args, probs=None, **kwargs):
    if tf.__version__ < "1.1":
        return tf_dists.Categorical(*args, p=probs, **kwargs)
    else:
        return tf_dists.Categorical(*args, probs=probs, **kwargs)


class Categorical(TensorFlowSelection):
    """ Don't use tf_dists.OneHotCategorical even if `one_hot` is True, because it doesn't work in tf versions <= 1.0.0 """

    def __init__(self, actions_dim, one_hot=True):
        self.params_dim = actions_dim
        self.actions_dim = actions_dim if one_hot else 1
        self.one_hot = one_hot

    def sample(self, utils, exploration):
        dist = self._dist(utils, exploration)
        sample = dist.sample()
        if self.one_hot:
            sample = tf.one_hot(sample, depth=int(utils.shape[-1]), axis=-1)
        else:
            sample = sample[..., None]
        return sample

    def log_probs(self, utils, actions, exploration):
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
        sample_shape = (self.actions_dim,) + (1,) * batch_rank
        sample = tf.reshape(tf.range(self.actions_dim), sample_shape)
        log_probs = dist.log_prob(sample)
        axis_perm = tuple(range(1, batch_rank+1)) + (0,)
        return tf.transpose(log_probs, perm=axis_perm)

    def entropy(self, utils, exploration):
        dist = self._dist(utils, exploration)
        return dist.entropy()[..., None]

    def kl(self, utils1, utils2, e1, e2=None):
        e2 = e1 if e2 is None else e2
        dist1 = self._dist(utils1, e1)
        dist2 = self._dist(utils2, e2)
        return tf_dists.kl(dist1, dist2)[..., None]

    def _dist(self, utils, exploration):
        raise Exception("NotImplemented")


class FixedCategorical(Categorical):
    def __init__(self, actions_dim, probs=None, logits=None, one_hot=True):
        assert (probs is None) != (logits is None)
        self.params_dim = 0
        self.actions_dim = actions_dim if one_hot else 1
        self.probs = probs
        self.logits = logits
        self.one_hot = one_hot

    def _dist(self, utils, exploration):
        return build_categorical_dist(logits=utils/exploration)


class Softmax(Categorical):
    def _dist(self, utils, exploration):
        return build_categorical_dist(logits=utils/exploration)


class EpsilonSoftmax(Categorical):
    """ Mixture between a softmax distribution and a uniform distribution.
        Weight of the uniform distribution is given by the exploration
        coefficient epsilon, and the softmax uses a temperature of 1.

    """
    def __init__(self, *args, softmax_temp=1.0, **kwargs):
        self.softmax_temp = softmax_temp
        super(EpsilonSoftmax, self).__init__(*args, **kwargs)

    def _dist(self, utils, epsilon):
        probs = (1 - epsilon) * tf.nn.softmax(utils) + epsilon / self.params_dim
        return build_categorical_dist(probs=probs)


class EpsilonGreedy(EpsilonSoftmax):
    def __init__(self, *args, **kwargs):
        super(EpsilonGreedy, self).__init__(*args, softmax_temp=0.1, **kwargs)
