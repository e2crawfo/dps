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

            log_prob, action, entropy, util, policy_state = policy.act(obs, policy_state, exploration)

            new_obs, reward, done, info = self.step(action)
            obs = new_obs
            if save_utils:
                rollouts.append(
                    obs, action, reward, done=float(done),
                    entropy=entropy, log_probs=log_prob, utils=util)
            else:
                rollouts.append(
                    obs, action, reward, done=float(done),
                    entropy=entropy, log_probs=log_prob)

            t += 1

            if render_mode is not None:
                self.render(mode=render_mode)

        exploration = exploration or tf.get_default_session().run(policy.exploration)
        rollouts.set_static('exploration', exploration)

        return rollouts

    def _pprint_rollouts(self, rollouts):
        pass

    def visualize(self, render_rollouts=None, **rollout_kwargs):
        rollouts = self.do_rollouts(**rollout_kwargs)
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
    scale_rewards = Param(True)

    def __init__(self, **kwargs):
        self._samplers = {}
        self._assert_defined('action_names')
        self._assert_defined('rb')
        self.mode = 'train'
        self._built = False
        super(TensorFlowEnv, self).__init__(**kwargs)

    @property
    def obs_shape(self):
        return (self.rb.visible_width,)

    def unpack_actions(self, actions):
        actions = tf.unstack(actions, axis=-1)
        actions = [a[..., None] for a in actions]
        return actions

    def _assert_defined(self, attr):
        assert getattr(self, attr) is not None, (
            "Subclasses of TensorFlowEnv must "
            "specify a value for attr {}.".format(attr))

    @abc.abstractmethod
    def start_episode(self):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_init(self):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_step(self, t, registers, action):
        raise Exception("NotImplemented")

    def get_sampler(self, policy):
        sampler = self._samplers.get(id(policy))
        if not sampler:
            with tf.name_scope("sampler_" + policy.display_name):
                n_rollouts_ph = tf.placeholder(tf.int32, (), name="n_rollouts_ph")
                T_ph = tf.placeholder(tf.int32, (), name="T_ph")

                initial_policy_state = policy.zero_state(n_rollouts_ph, tf.float32)
                _initial_registers = self.rb.new_array(n_rollouts_ph, lib='tf')
                initial_registers = self.build_init(_initial_registers)

                sampler_cell = SamplerCell(self, policy)

                t = timestep_tensor(n_rollouts_ph, T_ph)

                # Force build-step to be called in a safe environment for the first time.
                dummy_action = tf.zeros((n_rollouts_ph, self.actions_dim))
                self.build_step(t, initial_registers, dummy_action)

                _output = dynamic_rnn(
                    sampler_cell, t, initial_state=(initial_registers, initial_policy_state),
                    parallel_iterations=1, swap_memory=False, time_major=True)

                obs, hidden, done, utils, entropy, actions, log_probs, rewards, policy_states = _output[0]
                final_registers, final_policy_state = _output[1]

                if self.scale_rewards:
                    rewards /= tf.cast(T_ph, tf.float32)

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

            self._samplers[id(policy)] = (
                n_rollouts_ph, T_ph, initial_policy_state,
                initial_registers, output)

            sampler = self._samplers[id(policy)]

        return sampler

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train', save_utils=False):
        policy.set_mode(mode)
        self.set_mode(mode, n_rollouts)
        T = T or cfg.T

        n_rollouts_ph, T_ph, _, _, output = self.get_sampler(policy)
        n_rollouts, feed_dict = self.start_episode(n_rollouts)

        feed_dict.update({
            n_rollouts_ph: n_rollouts,
            T_ph: T,
        })

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

        if save_utils:
            rollouts = RolloutBatch(
                obs, actions, rewards, log_probs=log_probs, utils=utils, entropy=entropy,
                hidden=hidden, done=done, static=dict(exploration=exploration),
                metadata={'final_registers': final_registers})
        else:
            rollouts = RolloutBatch(
                obs, actions, rewards, log_probs=log_probs, entropy=entropy,
                hidden=hidden, done=done, static=dict(exploration=exploration),
                metadata={'final_registers': final_registers})
        return rollouts

    def _pprint_rollouts(self, rollouts):
        registers = np.concatenate([rollouts.obs, rollouts.hidden], axis=2)
        registers = np.concatenate(
            [registers, rollouts._metadata['final_registers'][np.newaxis, ...]],
            axis=0)
        actions = rollouts.a

        row_names = ['t=', ''] + self.action_names

        omit = set(self.rb.no_display)
        reg_ranges = {}

        T = actions.shape[0]

        for n, s in zip(self.rb.names, self.rb.shapes):
            if n in omit:
                continue

            row_names.append('')
            start = len(row_names)
            for k in range(s):
                row_names.append('{}[{}]'.format(n, k))
            end = len(row_names)
            reg_ranges[n] = (start, end)

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

            values = np.zeros((T+1, len(row_names)))
            for t in range(T):
                values[t, 0] = t
                offset = 2
                values[t, offset:offset+actions_dim] = actions[t, b, :]
                offset += actions_dim + 1

                for n, v in zip(self.rb.names, registers):
                    if n in omit:
                        continue
                    rr = reg_ranges[n]
                    values[t, rr[0]:rr[1]] = v[t, b, :]

                for k in other_keys:
                    v = rollouts[k]
                    _range = other_ranges[k]
                    values[t, _range[0]:_range[1]] = v[t, b, :]

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


class InternalEnv(TensorFlowEnv):
    target_shape = (1,)
    input_shape = None

    def start_episode(self, external_obs, external):
        return (
            external_obs.shape[0],
            {self.input_ph: external_obs,
             self.target_ph: external.y,
             self.is_training_ph: self.mode == 'train',
             self.is_testing_ph: self.mode == 'test'}
        )

    def build_placeholders(self, r):
        is_training_ph = getattr(self, 'is_training_ph', None)
        if is_training_ph is None:
            self.is_training_ph = tf.placeholder(tf.bool, (), name="is_training_ph")
            self.is_testing_ph = tf.placeholder(tf.bool, (), name="is_testing_ph")
            self.input_ph = tf.placeholder(tf.float32, (None,) + self.input_shape, name="input_ph")
            self.target_ph = tf.placeholder(tf.float32, (None,) + self.target_shape, name="target_ph")

    def build_reward(self, r):
        rewards = tf.cond(
            self.is_testing_ph,
            lambda: tf.fill((tf.shape(r)[0], 1), 0.0),
            lambda: -tf.cast(
                tf.reduce_sum(tf.abs(self.rb.get_output(r) - self.target_ph), axis=-1, keep_dims=True) > cfg.reward_window, tf.float32)
        )
        return rewards


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
    def completion(self):
        return self.external.completion

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train', save_utils=False):
        T = T or cfg.T
        (n_rollouts_ph, T_ph, initial_policy_state,
         initial_registers, output) = self.internal.get_sampler(policy)

        policy.set_mode(mode)
        self.external.set_mode(mode, n_rollouts)
        self.internal.set_mode(mode, n_rollouts)
        external_obs = self.external.reset()
        n_rollouts = external_obs.shape[0]

        final_registers = None
        final_policy_state = None

        e = 0

        external_rollouts = RolloutBatch()
        rollouts = RolloutBatch()
        external_done = False

        while not external_done:
            _, feed_dict = self.internal.start_episode(external_obs, self.external)

            if e > 0:
                feed_dict.update({
                    n_rollouts_ph: n_rollouts,
                    T_ph: T,
                    initial_registers: final_registers,
                })

                feed_dict.update(
                    tf.python.util.nest.flatten_dict_items(
                        {initial_policy_state: final_policy_state})
                )
            else:
                feed_dict.update({
                    n_rollouts_ph: n_rollouts,
                    T_ph: T,
                })

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

            external_action = self.rb.get_output(final_registers)
            new_external_obs, external_reward, external_done, external_info = \
                self.external.step(external_action)

            if self.final_reward:
                rewards[-1, ...] += external_reward

            if save_utils:
                rollout_batch = RolloutBatch(
                    obs, actions, rewards,
                    log_probs=log_probs, utils=utils,
                    entropy=entropy, hidden=hidden, done=done)
            else:
                rollout_batch = RolloutBatch(
                    obs, actions, rewards, log_probs=log_probs,
                    entropy=entropy, hidden=hidden, done=done)
            rollouts.extend(rollout_batch)

            external_info['length'] = rollouts.T
            external_rollouts.append(
                external_obs, external_action, external_reward,
                done=external_done, info=external_info)

            external_obs = new_external_obs

            e += 1

        rollouts.set_static('exploration', exploration)
        rollouts._metadata['final_registers'] = final_registers
        rollouts._metadata['external_rollouts'] = external_rollouts

        return rollouts

    def _pprint_rollouts(self, rollouts):
        registers = np.concatenate([rollouts.obs, rollouts.hidden], axis=2)
        registers = np.concatenate(
            [registers, rollouts._metadata['final_registers'][np.newaxis, ...]],
            axis=0)
        actions = rollouts.a

        external_rollouts = rollouts._metadata['external_rollouts']
        external_step_lengths = [i['length'] for i in external_rollouts.info]
        external_obs = external_rollouts.o

        total_internal_steps = sum(external_step_lengths)

        row_names = ['t=', 'i=', ''] + self.internal.action_names

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

            if getattr(self, 'obs_is_image', None):
                for e in external_obs:
                    print(image_to_string(e[b, :]))
            else:
                if np.product(external_obs[0].shape[1:]) < 40:
                    for e in external_obs:
                        print(e[b, :])

            values = np.zeros((total_internal_steps+1, len(row_names)))
            external_t, internal_t = 0, 0

            for i in range(total_internal_steps):
                values[i, 0] = external_t
                values[i, 1] = internal_t
                values[i, 3:3+actions_dim] = actions[i, b, :]

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

                if internal_t == external_step_lengths[external_t]:
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
