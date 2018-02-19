import abc
from future.utils import with_metaclass
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from dps import cfg
from dps.env import Env
from dps.rl import RolloutBatch
from dps.utils import Param
from dps.utils.tf import RNNCell


class SamplerCell(RNNCell):
    """ Each time an instance of this class is called, it samples one step of
        interaction between the provided (env,, policy) pair.

    """
    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self._state_size = (self.env.rb.width, self.policy.state_size)

        param_shape = tf.TensorShape(self.policy.param_shape)
        action_shape = tf.TensorShape(self.env.action_shape)

        self._output_size = (
            self.env.rb.visible_width, self.env.rb.hidden_width, 1,
            param_shape, 1, action_shape, 1, 1, self.policy.state_size)

    def __call__(self, t, state, scope=None):
        with tf.name_scope(scope or 'sampler_cell'):
            registers, policy_state = state

            with tf.name_scope('policy'):
                obs = self.env.rb.visible(registers)
                hidden = self.env.rb.hidden(registers)
                policy_output = self.policy(obs, policy_state)
                (log_prob, action, entropy, util), new_policy_state = policy_output

            with tf.name_scope('env_step'):
                done, reward, new_registers = self.env.build_step(t, registers, action)

            return (
                (obs, hidden, done, util, entropy, action,
                 log_prob, reward, policy_state),
                (new_registers, new_policy_state))

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        registers = self.env.rb.new_array(batch_size)
        policy_state = self.policy.zero_state(batch_size, dtype)
        return registers, policy_state


def timestep_tensor(batch_size, T):
    return tf.tile(tf.reshape(tf.range(T), (T, 1, 1)), (1, batch_size, 1))


class TensorFlowEnvMeta(abc.ABCMeta):
    def __init__(cls, *args, **kwargs):
        super(TensorFlowEnvMeta, cls).__init__(*args, **kwargs)
        if hasattr(cls.action_names, '__len__'):
            cls.action_shape = (len(cls.action_names),)


class TensorFlowEnv(with_metaclass(TensorFlowEnvMeta, Env)):
    """ An environment whose step method is implemented in TensorFlow,
        so that the rollout process can be run entirely within tensorflow. """
    rb = None
    action_names = None

    def __init__(self, **kwargs):
        self._samplers = {}
        self._assert_defined('action_names')
        self._assert_defined('rb')
        self._built = False
        self._placeholders_built = False
        super(TensorFlowEnv, self).__init__(**kwargs)

    @abc.abstractmethod
    def _build_placeholders(self):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def _make_feed_dict(self, *args, **kwargs):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_init(self, registers):
        """ Populate registers, providing initial values. """
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_step(self, t, registers, action):
        raise Exception("NotImplemented")

    @property
    def obs_shape(self):
        return (self.rb.visible_width,)

    def unpack_actions(self, actions):
        if self.action_sizes is None:
            return tf.split(actions, len(self.action_names), axis=-1)
        else:
            return tf.split(actions, self.action_sizes, axis=-1)

    def _assert_defined(self, attr):
        assert getattr(self, attr) is not None, (
            "Subclasses of TensorFlowEnv must specify a value for attr {}.".format(attr))

    def maybe_build_placeholders(self):
        """ Build placeholders for setting attributes of environment when sampling rollouts. """

        if not self._placeholders_built:
            self.n_rollouts = tf.placeholder(tf.int32, (), name="n_rollouts")
            self.T = tf.placeholder(tf.int32, (), name="T")
            self.mode = tf.placeholder(tf.string, (), name="mode")

            self.is_training = tf.equal(self.mode, 'train')
            self.is_testing = tf.equal(self.mode, 'test')

            self._build_placeholders()

            self._placeholders_built = True

    def make_feed_dict(self, n_rollouts, T, mode, *args, **kwargs):
        """ Create a feed dict to populate the placeholders created by `maybe_build_placeholders`. """
        feed_dict = self._make_feed_dict(n_rollouts, T, mode, *args, **kwargs)
        feed_dict.update({
            self.n_rollouts: n_rollouts,
            self.T: T,
            self.mode: mode,
        })
        return feed_dict

    def build_sampler(self, policy):
        sampler = self._samplers.get(id(policy))
        if not sampler:
            self.maybe_build_placeholders()

            with tf.name_scope("sampler_" + policy.display_name):
                initial_policy_state = policy.zero_state(self.n_rollouts, tf.float32)

                _initial_registers = self.rb.new_array(self.n_rollouts, lib='tf')
                initial_registers = self.build_init(_initial_registers)

                sampler_cell = SamplerCell(self, policy)

                t = timestep_tensor(self.n_rollouts, self.T)

                # Force build-step to be called in a safe environment for the first time.
                dummy_action = tf.zeros((self.n_rollouts,) + self.action_shape)
                self.build_step(t, initial_registers, dummy_action)

                _output = dynamic_rnn(
                    sampler_cell, t, initial_state=(initial_registers, initial_policy_state),
                    parallel_iterations=1, swap_memory=False, time_major=True)

                obs, hidden, done, utils, entropy, actions, log_probs, rewards, policy_states = _output[0]
                final_registers, final_policy_state = _output[1]

                output = dict(
                    obs=obs,
                    hidden=hidden,
                    done=done,
                    utils=utils,
                    entropy=entropy,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    policy_states=policy_states,
                    final_registers=final_registers,
                    final_policy_state=final_policy_state,
                    exploration=policy.exploration
                )

            self._samplers[id(policy)] = initial_policy_state, initial_registers, output
            sampler = self._samplers[id(policy)]

        return sampler

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train', save_utils=False):
        policy.set_mode(mode)

        T = T or cfg.T

        _, _, output = self.build_sampler(policy)
        feed_dict = self.make_feed_dict(n_rollouts, T, mode)

        if exploration is not None:
            feed_dict.update({policy.exploration: exploration})

        sess = tf.get_default_session()

        # sample rollouts
        obs, hidden, final_registers, utils, entropy, actions, log_probs, rewards, done, exploration = sess.run(
            [output['obs'],
             output['hidden'],
             output['final_registers'],
             output['utils'],
             output['entropy'],
             output['actions'],
             output['log_probs'],
             output['rewards'],
             output['done'],
             output['exploration']],
            feed_dict=feed_dict)

        kwargs = dict(
            log_probs=log_probs, utils=utils, entropy=entropy,
            hidden=hidden, done=done, static=dict(exploration=exploration),
            metadata=dict(final_registers=final_registers))

        if save_utils:
            kwargs['utils'] = utils

        return RolloutBatch(obs, actions, rewards, **kwargs)


class InternalEnv(TensorFlowEnv):
    """ A TensorFlowEnv that solves a supervised learning problem. Should be used inside a CompositeEnv. """

    target_shape = (1,)
    input_shape = None

    def _build_placeholders(self):
        self.input = tf.placeholder(tf.float32, (None,) + self.input_shape, name="input")
        self.target = tf.placeholder(tf.float32, (None,) + self.target_shape, name="target")

    def _make_feed_dict(self, n_rollouts, T, mode, inp, target):
        return {self.input: inp, self.target: target}

    def build_reward(self, registers, actions):
        rewards = tf.cond(
            self.is_testing,
            lambda: tf.zeros(tf.shape(registers)[:-1])[..., None],
            lambda: -tf.reduce_sum(
                tf.to_float(tf.abs(self.rb.get_output(registers) - self.target) > cfg.reward_window),
                axis=-1, keep_dims=True
            ),
        )
        rewards /= tf.to_float(self.T)
        return rewards

    def get_output(self, registers, actions):
        return self.rb.get_output(registers)


class CompositeEnv(Env):
    """ Combines an InternalEnv with another env which can be thought of as an external env,
        and acts as a loss function for the InternalEnv.

    """
    final_reward = Param(False)

    def __init__(self, external, internal):
        assert isinstance(internal, TensorFlowEnv)
        self.external, self.internal = external, internal
        self.rb = internal.rb

        self.obs_shape = (self.rb.visible_width,)
        self.action_shape = self.internal.action_shape

        self.reward_range = external.reward_range
        self.sampler = None

        super(CompositeEnv, self).__init__()

    @property
    def recorded_names(self):
        return self.external.recorded_names

    @property
    def action_sizes(self):
        return self.internal.action_sizes

    @property
    def action_names(self):
        return self.internal.action_names

    @property
    def n_discrete_actions(self):
        return self.internal.n_discrete_actions

    def build_trajectory_loss(self, actions, visible, hidden):
        return self.internal.build_trajectory_loss(actions, visible, hidden)

    @property
    def completion(self):
        return self.external.completion

    @property
    def has_differentiable_loss(self):
        return self.internal.has_differentiable_loss

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train', save_utils=False):
        T = T or cfg.T
        initial_policy_state, initial_registers, output = self.internal.build_sampler(policy)

        policy.set_mode(mode)
        self.external.set_mode(mode, n_rollouts)
        external_obs = self.external.reset()
        n_rollouts = external_obs.shape[0]

        final_registers = None
        final_policy_state = None

        e = 0

        external_rollouts = RolloutBatch()
        rollouts = RolloutBatch()
        external_done = False

        segment_lengths = []

        while not external_done:
            feed_dict = self.internal.make_feed_dict(n_rollouts, T, mode, external_obs, self.external.rl_y)

            if e > 0:
                feed_dict.update({initial_registers: final_registers})
                feed_dict.update(
                    tf.python.util.nest.flatten_dict_items(
                        {initial_policy_state: final_policy_state})
                )

            if exploration is not None:
                feed_dict.update({policy.exploration: exploration})

            sess = tf.get_default_session()

            (obs, hidden, done, final_registers, final_policy_state, utils, entropy,
             actions, log_probs, rewards, exploration) = sess.run(
                [output['obs'],
                 output['hidden'],
                 output['done'],
                 output['final_registers'],
                 output['final_policy_state'],
                 output['utils'],
                 output['entropy'],
                 output['actions'],
                 output['log_probs'],
                 output['rewards'],
                 output['exploration']],
                feed_dict=feed_dict)

            external_action = self.internal.get_output(final_registers, actions[-1, ...])
            new_external_obs, external_reward, external_done, external_info = self.external.step(external_action)

            if self.final_reward:
                rewards[-1, ...] += external_reward

            kwargs = dict(log_probs=log_probs, entropy=entropy, hidden=hidden, done=done)

            if save_utils:
                kwargs['utils'] = utils

            rollout_segment = RolloutBatch(obs, actions, rewards, **kwargs)

            rollouts.extend(rollout_segment)

            external_rollouts.append(
                external_obs, external_action, external_reward,
                done=external_done, info=external_info)

            external_info['length'] = rollout_segment.T
            segment_lengths.append(rollout_segment.T)

            for name in getattr(self.external, "recorded_names", []):
                rollouts._metadata[name] = np.mean(external_info[name])

            rollouts._metadata['description'] = external_obs

            external_obs = new_external_obs

            e += 1

        rollouts.set_static('exploration', exploration)
        rollouts._metadata['final_registers'] = final_registers
        rollouts._metadata['external_rollouts'] = external_rollouts
        rollouts._metadata['segment_lengths'] = segment_lengths

        return rollouts
