import abc
from future.utils import with_metaclass
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import pandas as pd
from tabulate import tabulate

import gym
from gym import Env as GymEnv
from gym.spaces import prng

from dps import cfg
from dps.rl import RolloutBatch
from dps.utils import Parameterized, Param, image_to_string
from dps.utils.tf import RNNCell


class BatchBox(gym.Space):
    """ A box that allows some dimensions to be unspecified at instance-creation time.

    Example usage:
    self.action_space = BatchBox(low=-10, high=10, shape=(None, 1))

    """
    def __init__(self, low, high, shape=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            shape = [1 if s is None else s for s in shape]
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)

    def sample(self):
        return prng.np_random.uniform(low=self.low, high=self.high, size=self.low.shape)

    def contains(self, x):
        return True
        # return (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    @property
    def shape(self):
        return (None,) + self.low.shape

    def __repr__(self):
        return "<BatchBox {}>".format(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


class Env(Parameterized, GymEnv, metaclass=abc.ABCMeta):
    has_differentiable_loss = False
    recorded_names = []
    action_sizes = None
    action_names = None

    def set_mode(self, mode, batch_size):
        """ Called at the beginning of `do_rollouts`.

        Sets the mode that the rollouts run under, as well as the number of rollouts to run.

        """
        assert mode in 'train val test'.split(), "Unknown mode: {}.".format(mode)
        self.mode = mode
        self.batch_size = batch_size

    def do_rollouts(
            self, policy, n_rollouts=None, T=None, exploration=None,
            mode='train', render_mode=None, save_utils=False):

        T = T or cfg.T

        policy.set_mode(mode)
        self.set_mode(mode, n_rollouts)
        obs = self.reset()
        batch_size = obs.shape[0]

        policy_state = policy.zero_state(batch_size, tf.float32)
        policy_state = tf.get_default_session().run(policy_state)

        rollouts = RolloutBatch()

        t = 0

        done = False
        while not done:
            if T is not None and t >= T:
                break

            log_prob, action, entropy, utils, policy_state = policy.act(obs, policy_state, exploration)

            new_obs, reward, done, info = self.step(action)
            obs = new_obs

            kwargs = dict(done=float(done), entropy=entropy, log_probs=log_prob)
            if save_utils:
                kwargs['utils'] = utils
            rollouts.append(obs, action, reward, **kwargs)

            t += 1

            if render_mode is not None:
                self.render(mode=render_mode)

        exploration = exploration or tf.get_default_session().run(policy.exploration)
        rollouts.set_static('exploration', exploration)

        return rollouts

    def _pprint_rollouts(self, rollouts):
        registers = np.concatenate([rollouts.obs, rollouts.hidden], axis=2)
        registers = np.concatenate(
            [registers, rollouts._metadata['final_registers'][None, ...]],
            axis=0)
        actions = rollouts.a

        segment_lengths = rollouts._metadata.get('segment_lengths', [registers.shape[0]])
        description = rollouts._metadata.get('description', None)

        total_steps = sum(segment_lengths)

        row_names = ['t=', 'i=', '']

        action_sizes = self.action_sizes or [1] * len(self.action_names)
        action_ranges = {}

        for n, s in zip(self.action_names, action_sizes):
            start = len(row_names)
            if s == 1:
                row_names.append(n)
            else:
                for k in range(s):
                    row_names.append('{}[{}]'.format(n, k))
            end = len(row_names)
            action_ranges[n] = (start, end)

        omit = set(self.rb.no_display)
        reg_ranges = {}

        for name, shape in zip(self.rb.names, self.rb.shapes):
            if name in omit:
                continue

            row_names.append('')
            start = len(row_names)
            row_names.extend(['{}[{}]'.format(name, k) for k in range(shape)])
            end = len(row_names)
            reg_ranges[name] = (start, end)

        other_ranges = {}
        other_keys = sorted(rollouts.keys())
        other_keys.remove('obs')
        other_keys.remove('actions')
        other_keys.remove('hidden')

        for k in other_keys:
            row_names.append('')

            start = len(row_names)
            row_names.extend(['{}[{}]'.format(k, i) for i in range(rollouts[k].shape[2])])
            end = len(row_names)
            other_ranges[k] = (start, end)

        n_timesteps, batch_size, actions_dim = actions.shape

        registers = self.rb.as_tuple(registers)

        for b in range(batch_size):
            print("\nElement {} of batch ".format(b) + "-" * 40)

            if description is not None:
                if getattr(self, 'obs_is_image', None):
                    print(image_to_string(description[b, ...]))
                else:
                    if np.product(description.shape[1:]) < 40:
                        print(description[b, ...])

            values = np.zeros((total_steps+1, len(row_names)))
            external_t, internal_t = 0, 0

            for i in range(total_steps):
                values[i, 0] = external_t
                values[i, 1] = internal_t

                offset = 0
                for name, size in zip(self.action_names, action_sizes):
                    ar = action_ranges[name]
                    values[i, ar[0]:ar[1]] = actions[i, b, offset:offset + size]
                    offset += size

                for name, v in zip(self.rb.names, registers):
                    if name in omit:
                        continue
                    rr = reg_ranges[name]
                    values[i, rr[0]:rr[1]] = v[i, b, :]

                for k in other_keys:
                    v = rollouts[k]
                    _range = other_ranges[k]
                    values[i, _range[0]:_range[1]] = v[i, b, :]

                internal_t += 1

                if internal_t == segment_lengths[external_t]:
                    external_t += 1
                    internal_t = 0

            # Print final values for the registers
            for n, v in zip(self.rb.names, registers):
                if n in omit:
                    continue
                rr = reg_ranges[n]
                values[-1, rr[0]:rr[1]] = v[-1, b, :]

            values = pd.DataFrame(values.T)
            values.insert(0, 'name', row_names)
            values = values.set_index('name')
            print(tabulate(values, headers='keys', tablefmt='fancy_grid'))

    def visualize(self, render_rollouts=None, **rollout_kwargs):
        rollouts = self.do_rollouts(**rollout_kwargs, save_utils=cfg.save_utils)
        self._pprint_rollouts(rollouts)
        if render_rollouts is not None and (cfg.save_plots or cfg.show_plots):
            render_rollouts(self, rollouts)


class SamplerCell(RNNCell):
    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self._state_size = (self.env.rb.width, self.policy.state_size)

        self._output_size = (
            self.env.rb.visible_width, self.env.rb.hidden_width, 1,
            self.policy.params_dim, 1, self.env.actions_dim, 1, 1, self.policy.state_size)

    def __call__(self, t, state, scope=None):
        with tf.name_scope(scope or 'sampler_cell'):
            registers, policy_state = state

            with tf.name_scope('policy'):
                obs = self.env.rb.visible(registers)
                hidden = self.env.rb.hidden(registers)
                (log_prob, action, entropy, util), new_policy_state = self.policy(obs, policy_state)

            with tf.name_scope('env_step'):
                done, reward, new_registers = self.env.build_step(t, registers, action)

            return (
                (obs, hidden, done, util, entropy, action, log_prob, reward, policy_state),
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
            cls.actions_dim = len(cls.action_names)


class TensorFlowEnv(with_metaclass(TensorFlowEnvMeta, Env)):
    rb = None
    action_names = None

    def __init__(self, **kwargs):
        self._samplers = {}
        self._assert_defined('action_names')
        self._assert_defined('rb')
        self.mode = 'train'
        self._built = False
        self._placeholders_built = False
        super(TensorFlowEnv, self).__init__(**kwargs)

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
            "Subclasses of TensorFlowEnv must "
            "specify a value for attr {}.".format(attr))

    def make_feed_dict(self, mode, n_rollouts, T):
        """ Create a feed-dict for sampling the rollouts. """
        return {
            self.mode_ph: mode,
            self.n_rollouts_ph: n_rollouts,
            self.T_ph: T,
        }

    def maybe_build_placeholders(self):
        if not self._placeholders_built:
            self.n_rollouts_ph = tf.placeholder(tf.int32, (), name="n_rollouts_ph")
            self.T_ph = tf.placeholder(tf.int32, (), name="T_ph")
            self.mode_ph = tf.placeholder(tf.string, (), name="mode_ph")

            self.n_rollouts = tf.identity(self.n_rollouts_ph, name="n_rollouts")
            self.T = tf.identity(self.T_ph, name="T")
            self.mode = tf.identity(self.mode_ph, name="mode")

            self.is_training = tf.equal(self.mode_ph, 'train')
            self.is_testing = tf.equal(self.mode_ph, 'test')
            self._placeholders_built = True

    @abc.abstractmethod
    def build_init(self, registers):
        """ Populate registers, providing initial values. """
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_step(self, t, registers, action):
        raise Exception("NotImplemented")

    def get_sampler(self, policy):
        sampler = self._samplers.get(id(policy))
        if not sampler:
            self.maybe_build_placeholders()

            with tf.name_scope("sampler_" + policy.display_name):
                initial_policy_state = policy.zero_state(self.n_rollouts_ph, tf.float32)

                _initial_registers = self.rb.new_array(self.n_rollouts_ph, lib='tf')
                initial_registers = self.build_init(_initial_registers)

                sampler_cell = SamplerCell(self, policy)

                t = timestep_tensor(self.n_rollouts_ph, self.T_ph)

                # Force build-step to be called in a safe environment for the first time.
                dummy_action = tf.zeros((self.n_rollouts_ph, self.actions_dim))
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

    def do_rollouts(
            self, policy, n_rollouts=None, T=None, exploration=None,
            mode='train', save_utils=False):

        policy.set_mode(mode)

        T = T or cfg.T

        _, _, output = self.get_sampler(policy)
        feed_dict = self.make_feed_dict(mode, n_rollouts, T)

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
    """ A TensorFlowEnv that solves a supervised learning problem. """

    target_shape = (1,)
    input_shape = None

    def make_feed_dict(self, mode, n_rollouts, T, inp, targets):
        feed_dict = super(InternalEnv, self).make_feed_dict(mode, n_rollouts, T)
        feed_dict.update({
            self.input_ph: inp,
            self.target_ph: targets
        })
        return feed_dict

    def maybe_build_placeholders(self):
        if getattr(self, 'input_ph', None) is None:
            self.input_ph = tf.placeholder(
                tf.float32, (None,) + self.input_shape, name="input_ph")
            self.target_ph = tf.placeholder(
                tf.float32, (None,) + self.target_shape, name="target_ph")

            self.input = tf.identity(self.input_ph, name="input")
            self.target = tf.identity(self.target_ph, name="target")

        super(InternalEnv, self).maybe_build_placeholders()

    def build_reward(self, registers, actions):
        rewards = tf.cond(
            self.is_testing,
            lambda: tf.zeros(tf.shape(registers)[:-1])[..., None],
            lambda: -tf.reduce_sum(
                tf.to_float(tf.abs(self.rb.get_output(registers) - self.target_ph) > cfg.reward_window),
                axis=-1, keep_dims=True
            ),
        )
        rewards /= tf.to_float(self.T)
        return rewards

    def get_output(self, registers, actions):
        return self.rb.get_output(registers)


class CompositeEnv(Env):
    final_reward = Param(False)

    def __init__(self, external, internal):
        assert isinstance(internal, TensorFlowEnv)
        self.external, self.internal = external, internal
        self.rb = internal.rb

        self.obs_shape = (self.rb.visible_width,)
        self.actions_dim = self.internal.actions_dim

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
        initial_policy_state, initial_registers, output = self.internal.get_sampler(policy)

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
            feed_dict = self.internal.make_feed_dict(
                mode, n_rollouts, T, external_obs, self.external.y)

            if e > 0:
                feed_dict.update({
                    initial_registers: final_registers,
                })

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
