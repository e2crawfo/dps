import abc
from future.utils import with_metaclass
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import pandas as pd
from tabulate import tabulate

import gym
from gym import Env as GymEnv
from gym.utils import seeding
from gym.spaces import prng

from dps import cfg
from dps.rl import RolloutBatch


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
        return self.low.shape

    def __repr__(self):
        return "<BatchBox {}>".format(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


class Env(with_metaclass(abc.ABCMeta, GymEnv)):
    def set_mode(self, mode, batch_size):
        assert mode in 'train train_eval val'.split(), "Unknown mode: {}.".format(mode)
        self.mode = mode
        self.batch_size = batch_size

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train'):
        T = T or cfg.T
        start_time = time.time()
        self.set_mode(mode, n_rollouts)
        obs = self.reset()
        batch_size = obs.shape[0]

        policy_state = policy.zero_state(batch_size, tf.float32)
        policy_state = tf.get_default_session().run(policy_state)

        done = False
        while not done:
            action, policy_state = policy.act(obs, policy_state, exploration)
            new_obs, reward, done, info = self.step(action)
            obs = new_obs

        print("Took {} seconds to do {} rollouts.".format(time.time() - start_time, n_rollouts))

    def _pprint_rollouts(**kwargs):
        raise Exception("NotImplemented.")

    def visualize(self, render_rollouts=None, **rollout_kwargs):
        rollouts = self.do_rollouts(**rollout_kwargs)
        self._pprint_rollouts(rollouts)
        if render_rollouts is not None and (cfg.save_display or cfg.display):
            render_rollouts(self, rollouts)


class RegressionDataset(object):
    def __init__(self, x, y, shuffle=True):
        self.x = x
        self.y = y
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def obs_shape(self):
        return self.x.shape[1:]

    @property
    def n_examples(self):
        return self.x.shape[0]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def completion(self):
        return self.epochs_completed + self.index_in_epoch / self.n_examples

    def next_batch(self, batch_size=None, advance=True):
        """ Return the next ``batch_size`` examples from this data set.

        If ``batch_size`` not specified, return rest of the examples in the current epoch.

        """
        start = self._index_in_epoch

        if batch_size is None:
            batch_size = self.n_examples - start
        elif batch_size > self.n_examples:
            raise Exception("Too few examples ({}) to satisfy batch size of {}.".format(self.n_examples, batch_size))

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and self.shuffle:
            perm0 = np.arange(self.n_examples)
            np.random.shuffle(perm0)
            self._x = self.x[perm0]
            self._y = self.y[perm0]

        if start + batch_size >= self.n_examples:
            # Finished epoch

            # Get the remaining examples in this epoch
            x_rest_part = self._x[start:]
            y_rest_part = self._y[start:]

            # Shuffle the data
            if self.shuffle and advance:
                perm = np.arange(self.n_examples)
                np.random.shuffle(perm)
                self._x = self.x[perm]
                self._y = self.y[perm]

            # Start next epoch
            end = batch_size - len(x_rest_part)
            x_new_part = self._x[:end]
            y_new_part = self._y[:end]
            x = np.concatenate((x_rest_part, x_new_part), axis=0)
            y = np.concatenate((y_rest_part, y_new_part), axis=0)

            if advance:
                self._index_in_epoch = end
                self._epochs_completed += 1
        else:
            # Middle of epoch
            end = start + batch_size
            x, y = self._x[start:end], self._y[start:end]

            if advance:
                self._index_in_epoch = end

        return x, y


class RegressionEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, train, val):
        self.train, self.val = train, val
        self.datasets = {
            'train': self.train,
            'train_eval': self.train,
            'val': self.val,
        }

        self.n_actions = self.train.y.shape[1]
        self.action_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, self.n_actions))

        self.obs_shape = self.train.x.shape[1:]
        self.observation_space = BatchBox(low=-np.inf, high=np.inf, shape=(None,) + self.obs_shape)

        self.reward_range = (-np.inf, 0)

        self.mode = 'train'
        self.batch_size = None

        self.action_ph, self.loss, self.target_ph = None, None, None

        self.reset()

    def __str__(self):
        return "<RegressionEnv train={} val={}>".format(self.train, self.val)

    def next_batch(self, batch_size=None, mode='train'):
        advance = mode == 'train'
        return self.datasets[mode].next_batch(batch_size=batch_size, advance=advance)

    def build_loss(self, actions, targets):
        return tf.reduce_mean(tf.abs(actions - targets), axis=-1, keep_dims=True)
        # return tf.reduce_mean((actions - targets)**2, axis=-1, keep_dims=True)

    def build_reward(self, actions, targets):
        abs_error = tf.reduce_sum(tf.abs(actions - targets), axis=-1, keep_dims=True)
        return -tf.cast(abs_error > cfg.reward_window, tf.float32)

    @property
    def completion(self):
        return self.train.completion

    def _step(self, action):
        assert self.action_space.contains(action), (
            "{} ({}) is not a valid action for env {}.".format(action, type(action), self))
        self.t += 1

        assert self.y.shape == action.shape
        obs = np.zeros(self.x.shape)

        if self.action_ph is None:
            self.target_ph = tf.placeholder(tf.float32, (None, self.n_actions))
            self.action_ph = tf.placeholder(tf.float32, (None, self.n_actions))
            self.reward = self.build_reward(self.action_ph, self.target_ph)

        sess = tf.get_default_session()
        reward = sess.run(self.reward, {self.action_ph: action, self.target_ph: self.y})

        done = True
        info = {"y": self.y}
        return obs, reward, done, info

    def _reset(self):
        self.t = 0

        advance = self.mode == 'train'
        self.x, self.y = self.datasets[self.mode].next_batch(self.batch_size, advance=advance)
        return self.x

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class SamplerCell(RNNCell):
    def __init__(self, env, policy, static_inp):
        self.env, self.policy = env, policy
        self._state_size = (self.env.rb.width, self.policy.state_size)

        self._output_size = (self.env.rb.width, self.policy.n_params, 1, self.env.n_actions, 1, 1, self.policy.state_size)

        self.static_inp = static_inp

    def __call__(self, t, state, scope=None):
        with tf.name_scope(scope or 'sampler_cell'):
            registers, policy_state = state

            with tf.name_scope('policy'):
                obs = self.env.rb.visible(registers)
                utils, new_policy_state = self.policy.build_update(obs, policy_state)
                action = self.policy.build_sample(utils)
                log_prob = self.policy.build_log_prob(utils, action)
                entropy = self.policy.build_entropy(utils)

            with tf.name_scope('env_step'):
                reward, new_registers = self.env.build_step(t, registers, action, self.static_inp)

            return (registers, utils, entropy, action, log_prob, reward, policy_state), (new_registers, new_policy_state)

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
            cls.n_actions = len(cls.action_names)


class TensorFlowEnv(with_metaclass(TensorFlowEnvMeta, Env)):
    rb = None
    action_names = None

    def __init__(self):
        self._samplers = {}
        self._assert_defined('action_names')
        self._assert_defined('rb')

    @property
    def obs_shape(self):
        return (self.rb.visible_width,)

    def _assert_defined(self, attr):
        assert getattr(self, attr) is not None, (
            "Subclasses of TensorFlowEnv must "
            "specify a value for attr {}.".format(attr))

    @abc.abstractmethod
    def static_inp_type_and_shape(self):
        # For a single example - so doesn't include batch size.
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_init(self):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_step(self, t, registers, action, static_inp):
        # return reward, new_registers
        raise Exception("NotImplemented")

    def get_sampler(self, policy):
        sampler = self._samplers.get(id(policy))
        if not sampler:
            n_rollouts_ph = tf.placeholder(tf.int32, ())
            T_ph = tf.placeholder(tf.int32, ())

            initial_policy_state = policy.zero_state(n_rollouts_ph, tf.float32)
            initial_registers = self.rb.new_array(n_rollouts_ph, lib='tf')

            si_type, si_shape = self.static_inp_type_and_shape()
            static_inp_ph = tf.placeholder(si_type, (None,) + si_shape)

            sampler_cell = SamplerCell(self, policy, static_inp_ph)
            with tf.name_scope("sampler"):
                initial_registers = self.build_init(initial_registers, static_inp_ph)
                t = timestep_tensor(n_rollouts_ph, T_ph)

                _output = dynamic_rnn(
                    sampler_cell, t, initial_state=(initial_registers, initial_policy_state),
                    parallel_iterations=1, swap_memory=False, time_major=True)

                registers, utils, entropy, actions, log_probs, rewards, policy_states = _output[0]
                final_registers, final_policy_state = _output[1]

                output = dict(
                    registers=registers,
                    utils=utils,
                    entropy=entropy,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    policy_states=policy_states,
                    final_registers=final_registers,
                    final_policy_state=final_policy_state)

            self._samplers[id(policy)] = (
                n_rollouts_ph, T_ph, initial_policy_state, initial_registers, static_inp_ph, output)

            sampler = self._samplers[id(policy)]

        return sampler

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train'):
        self.set_mode(mode, n_rollouts)
        T = T or cfg.T
        static_inp = self.make_static_input(n_rollouts)
        n_rollouts = static_inp.shape[0]

        n_rollouts_ph, T_ph, _, _, static_inp_ph, output = self.get_sampler(policy)

        feed_dict = {
            n_rollouts_ph: n_rollouts,
            T_ph: T,
            static_inp_ph: static_inp,
        }

        if exploration is not None:
            feed_dict.update({policy.exploration: exploration})

        sess = tf.get_default_session()

        # sample rollouts
        registers, final_registers, utils, entropy, actions, log_probs, rewards = sess.run(
            [output['registers'],
             output['final_registers'],
             output['utils'],
             output['entropy'],
             output['actions'],
             output['log_probs'],
             output['rewards']],
            feed_dict=feed_dict)

        rollouts = RolloutBatch(
            registers, actions, rewards, log_probs=log_probs, utils=utils, entropy=entropy,
            metadata={'final_registers': final_registers})
        return rollouts

    def _pprint_rollouts(self, rollouts):
        registers = np.concatenate(
            [rollouts.o, rollouts._metadata['final_registers'][np.newaxis, ...]],
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

        for k in other_keys:
            row_names.append('')

            start = len(row_names)
            row_names.extend(['{}[{}]'.format(k, i) for i in range(rollouts[k].shape[2])])
            end = len(row_names)
            other_ranges[k] = (start, end)

        n_timesteps, batch_size, n_actions = actions.shape

        registers = self.rb.as_tuple(registers)

        for b in range(batch_size):
            print("\nElement {} of batch ".format(b) + "-" * 40)

            values = np.zeros((T+1, len(row_names)))
            for t in range(T):
                values[t, 0] = t
                offset = 2
                values[t, offset:offset+n_actions] = actions[t, b, :]
                offset += n_actions + 1

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


class CompositeEnv(Env):
    def __init__(self, external, internal):
        super(CompositeEnv, self).__init__()
        assert isinstance(internal, TensorFlowEnv)
        self.external, self.internal = external, internal
        self.rb = internal.rb

        self.n_actions = self.internal.n_actions
        self.action_space = BatchBox(low=0.0, high=1.0, shape=(None, self.n_actions))

        self.obs_shape = (self.rb.visible_width,)
        self.observation_space = BatchBox(low=-np.inf, high=np.inf, shape=(None,)+self.obs_shape)

        self.reward_range = external.reward_range
        self.sampler = None

    @property
    def completion(self):
        return self.external.completion

    def do_rollouts(
            self, policy, n_rollouts=None, T=None, exploration=None, mode='train'):

        T = T or cfg.T
        (n_rollouts_ph, T_ph, initial_policy_state,
         initial_registers, static_inp_ph, output) = self.internal.get_sampler(policy)

        self.external.set_mode(mode, n_rollouts)
        external_obs = self.external.reset()
        n_rollouts = external_obs.shape[0]

        final_registers = None
        final_policy_state = None

        done = False
        e = 0

        external_rollouts = RolloutBatch()
        rollouts = RolloutBatch()

        while not done:
            if e > 0:
                feed_dict = {
                    n_rollouts_ph: n_rollouts,
                    T_ph: T,
                    initial_registers: final_registers,
                    static_inp_ph: external_obs,
                }

                feed_dict.update(
                    tf.python.util.nest.flatten_dict_items(
                        {initial_policy_state: final_policy_state})
                )
            else:
                feed_dict = {
                    n_rollouts_ph: n_rollouts,
                    T_ph: T,
                    static_inp_ph: external_obs,
                }

            if exploration is not None:
                feed_dict.update({policy.exploration: exploration})

            sess = tf.get_default_session()

            (registers, final_registers, final_policy_state, utils, entropy,
             actions, log_probs, rewards) = sess.run(
                [output['registers'],
                 output['final_registers'],
                 output['final_policy_state'],
                 output['utils'],
                 output['entropy'],
                 output['actions'],
                 output['log_probs'],
                 output['rewards']],
                feed_dict=feed_dict)

            external_action = self.rb.get_output(final_registers)
            new_external_obs, external_reward, done, i = \
                self.external.step(external_action)

            rewards[-1, :, :] += external_reward

            rollouts.extend(
                RolloutBatch(
                    registers, actions, rewards,
                    log_probs=log_probs, utils=utils, entropy=entropy))

            external_rollouts.append(
                external_obs, external_action, external_reward,
                info=dict(length=len(rollouts)))

            external_obs = new_external_obs

            e += 1

        rollouts._metadata['final_registers'] = final_registers
        rollouts._metadata['external_rollouts'] = external_rollouts

        return rollouts

    def _pprint_rollouts(self, rollouts):
        registers = np.concatenate(
            [rollouts.o, rollouts._metadata['final_registers'][np.newaxis, ...]],
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

        for k in other_keys:
            row_names.append('')

            start = len(row_names)
            row_names.extend(['{}[{}]'.format(k, i) for i in range(rollouts[k].shape[2])])
            end = len(row_names)
            other_ranges[k] = (start, end)

        n_timesteps, batch_size, n_actions = actions.shape

        registers = self.rb.as_tuple(registers)

        for b in range(batch_size):
            print("\nElement {} of batch ".format(b) + "-" * 40)

            if external_obs[0].shape[-1] < 40:
                print("External observations: ")
                for i, e in enumerate(external_obs):
                    print("{}: {}".format(i, e[b, :]))

            values = np.zeros((total_internal_steps+1, len(row_names)))
            external_t, internal_t = 0, 0

            for i in range(total_internal_steps):
                values[i, 0] = external_t
                values[i, 1] = internal_t
                values[i, 3:3+n_actions] = actions[i, b, :]

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
