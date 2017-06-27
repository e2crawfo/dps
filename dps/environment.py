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
from dps.utils import uninitialized_variables_initializer


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
    def set_mode(self, kind, batch_size):
        assert kind in ['train', 'val', 'test'], "Unknown kind {}.".format(kind)
        self._kind = kind
        self._batch_size = batch_size

    def do_rollouts(self, alg, policy, n_rollouts=None, T=None, mode='train'):
        T = T or cfg.T
        start_time = time.time()
        self.set_mode(mode, n_rollouts)
        obs = self.reset()
        batch_size = obs.shape[0]

        alg.start_episode()
        policy_state = policy.zero_state(batch_size, tf.float32)
        policy_state = tf.get_default_session().run(policy_state)

        done = False
        while not done:
            action, policy_state = policy.act(obs, policy_state)
            new_obs, reward, done, info = self.step(action)

            alg.remember(obs, action, reward)
            obs = new_obs

        alg.end_episode()

        print("Took {} seconds to do {} rollouts.".format(time.time() - start_time, n_rollouts))


class RegressionDataset(object):
    def __init__(self, x, y, for_eval=False, shuffle=True):
        self.x = x
        self.y = y
        self.for_eval = for_eval
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0

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

    def next_batch(self, batch_size=None):
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

        # Go to the next epoch
        if start + batch_size >= self.n_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_n_examples = self.n_examples - start
            x_rest_part = self._x[start:self.n_examples]
            y_rest_part = self._y[start:self.n_examples]

            if self.for_eval:
                self._index_in_epoch = 0
                return x_rest_part, y_rest_part
            else:
                # Shuffle the data
                if self.shuffle:
                    perm = np.arange(self.n_examples)
                    np.random.shuffle(perm)
                    self._x = self.x[perm]
                    self._y = self.y[perm]

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_n_examples
                end = self._index_in_epoch
                x_new_part = self._x[start:end]
                y_new_part = self._y[start:end]
                x = np.concatenate((x_rest_part, x_new_part), axis=0)
                y = np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            x, y = self._x[start:end], self._y[start:end]

        return x, y


class RegressionEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, train, val, test):
        self.train, self.val, self.test = train, val, test

        self.n_actions = self.train.y.shape[1]
        self.action_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, self.n_actions))

        self.obs_dim = self.train.x.shape[1]
        self.observation_space = BatchBox(
            low=-np.inf, high=np.inf, shape=(None, self.obs_dim))

        self.reward_range = (-np.inf, 0)

        self._kind = 'train'
        self._batch_size = None

        self.action_ph, self.loss, self.target_ph = None, None, None

        self.reset()

    def __str__(self):
        return "<RegressionEnv train={} val={} test={}>".format(self.train, self.val, self.test)

    @property
    def completion(self):
        return self.train.completion

    def build_loss(self, actions, idx=0):
        target_ph = tf.placeholder(tf.float32, shape=actions.shape, name='target')
        error = tf.reduce_sum(tf.abs(actions - target_ph), axis=-1, keep_dims=True)
        loss = tf.cast(error > cfg.reward_window, tf.float32)
        return loss, target_ph

    def _step(self, action):
        assert self.action_space.contains(action), (
            "{} ({}) is not a valid action for env {}.".format(action, type(action), self))
        self.t += 1

        assert self.y.shape == action.shape
        obs = np.zeros(self.x.shape)

        if self.action_ph is None:
            self.action_ph = tf.placeholder(tf.float32, (None, self.n_actions))
            self.loss, self.target_ph = self.build_loss(self.action_ph)

        sess = tf.get_default_session()
        reward = -sess.run(self.loss, {self.action_ph: action, self.target_ph: self.y})

        done = True
        info = {"y": self.y}
        return obs, reward, done, info

    def _reset(self):
        self.t = 0

        dataset = getattr(self, self._kind)
        self.x, self.y = dataset.next_batch(self._batch_size)
        return self.x

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class SamplerCell(RNNCell):
    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self._state_size = (self.env.rb.width, self.policy.state_size)

        self._output_size = (
            self.env.rb.width,
            self.env.n_actions,
            1,
            self.policy.state_size)

        self._static_inp = None

    def set_static_input(self, static_inp):
        """ Provide a batched tensor that is the same every time step,
            and which can be accessed by the environment.

        """
        self._static_inp = static_inp

    def __call__(self, t, state, scope=None):
        with tf.name_scope(scope or 'sampler_cell'):
            registers, policy_state = state

            with tf.name_scope('policy'):
                obs = self.env.rb.visible(registers)
                action, _, new_policy_state = self.policy.build_sample(obs, policy_state)

            with tf.name_scope('env_step'):
                reward, new_registers = self.env.build_step(t, registers, action, self._static_inp)

            return (registers, action, reward, policy_state), (new_registers, new_policy_state)

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


class Sampler(object):
    """ An object that stores an env and a policy.
        Its build function creates operations that sample rollouts using the env and policy.

        Calling its __call__ method builds the graph.

    """
    def __init__(self, env, policy, scope=None):
        self.env, self.policy, self.scope = env, policy, scope
        self._sampler_cell = None

    def __str__(self):
        return "<Sampler - env={}, policy={}, scope={}>".format(self.env, self.policy)

    @property
    def sampler_cell(self):
        if self._sampler_cell is None:
            self._sampler_cell = SamplerCell(self.env, self.policy)
        return self._sampler_cell

    def __call__(self, n_rollouts, T, initial_policy_state, initial_registers, static_inp):
        with tf.name_scope(self.scope or "sampler"):
            self.sampler_cell.set_static_input(static_inp)

            initial_registers = self.env.build_init(initial_registers, static_inp)

            t = timestep_tensor(n_rollouts, T)

            _output = dynamic_rnn(
                self.sampler_cell, t, initial_state=(initial_registers, initial_policy_state),
                parallel_iterations=1, swap_memory=False, time_major=True)

            (
                (registers, actions, rewards, policy_states),
                (final_registers, final_policy_state)) = _output

            output = dict(
                registers=registers,
                actions=actions,
                rewards=rewards,
                policy_states=policy_states,
                final_registers=final_registers,
                final_policy_state=final_policy_state)

        return output


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
    def obs_dim(self):
        return self.rb.visible_width

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
            sampler = Sampler(self, policy)
            n_rollouts_ph = tf.placeholder(tf.int32, ())
            T_ph = tf.placeholder(tf.int32, ())

            initial_policy_state = policy.zero_state(n_rollouts_ph, tf.float32)
            initial_registers = self.rb.new_array(n_rollouts_ph, lib='tf')

            si_type, si_shape = self.static_inp_type_and_shape()
            static_inp_ph = tf.placeholder(si_type, (None,) + si_shape)

            outputs = sampler(n_rollouts_ph, T_ph, initial_policy_state, initial_registers, static_inp_ph)

            self._samplers[id(policy)] = (
                sampler, n_rollouts_ph, T_ph,
                initial_policy_state, initial_registers,
                static_inp_ph, outputs)
            sampler = self._samplers[id(policy)]

        return sampler

    def do_rollouts(self, alg, policy, n_rollouts=None, T=None, mode='train'):
        T = T or cfg.T
        sampler = self.get_sampler(policy)

        self.set_mode(mode)
        static_inp = self.get_static_input(n_rollouts)

        sampler, n_rollouts_ph, T_ph, _, _, static_inp_ph, output = self.get_sampler(policy)

        feed_dict = {
            n_rollouts_ph: n_rollouts,
            T_ph: T,
            static_inp_ph: static_inp,
            alg.is_training: mode == 'train'
        }

        sess = tf.get_default_session()

        # sample rollouts
        registers, final_registers, actions, rewards = sess.run(
            [output['registers'],
             output['final_registers'],
             output['actions'],
             output['rewards']],
            feed_dict=feed_dict)

        # record rollouts
        alg.start_episode()

        for t, (o, a, r) in enumerate(zip(self.rb.visible(registers), actions, rewards)):
            alg.remember(o, a, r)

        alg.end_episode()

        info = []

        return registers, actions, rewards, info


class DummyAlg(object):
    def start_episode(self):
        pass

    def remember(self, o, a, r):
        pass

    def end_episode(self):
        pass

    @property
    def is_training(self):
        return tf.constant(np.array(False))


class CompositeEnv(Env):
    def __init__(self, external, internal):
        super(CompositeEnv, self).__init__()
        assert isinstance(internal, TensorFlowEnv)
        self.external, self.internal = external, internal
        self.rb = internal.rb

        self.n_actions = self.internal.n_actions
        self.action_space = BatchBox(low=0.0, high=1.0, shape=(None, self.n_actions))

        self.obs_dim = self.rb.visible_width
        self.observation_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, self.obs_dim))

        self.reward_range = external.reward_range
        self.sampler = None

    @property
    def completion(self):
        return self.external.completion

    def do_rollouts(self, alg, policy, n_rollouts=None, T=None, mode='train'):
        T = T or cfg.T
        (sampler, n_rollouts_ph, T_ph,
         initial_policy_state, initial_registers,
         static_inp_ph, output) = self.internal.get_sampler(policy)

        self.external.set_mode(mode, n_rollouts)
        external_obs = self.external.reset()
        n_rollouts = external_obs.shape[0]

        alg.start_episode()

        final_registers = None
        final_policy_state = None

        done = False
        e = 0
        registers, actions, rewards, info = [], [], [], []

        while not done:
            if e > 0:
                feed_dict = {
                    n_rollouts_ph: n_rollouts,
                    T_ph: T,
                    initial_registers: final_registers,
                    static_inp_ph: external_obs,
                    alg.is_training: mode == 'train'
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
                    alg.is_training: mode == 'train'
                }

            sess = tf.get_default_session()

            _registers, final_registers, final_policy_state, _actions, _rewards = sess.run(
                [output['registers'],
                 output['final_registers'],
                 output['final_policy_state'],
                 output['actions'],
                 output['rewards']],
                feed_dict=feed_dict)

            registers.append(_registers)
            actions.append(_actions)
            rewards.append(_rewards)

            _info = dict(external_obs=external_obs)

            external_action = self.rb.get_output(final_registers)
            external_obs, external_reward, done, i = self.external.step(external_action)

            _rewards[-1, :, :] += external_reward

            _info.update(
                i,
                external_action=external_action,
                external_reward=external_reward,
                done=done,
                length=_actions.shape[0])

            info.append(_info)

            # record the trajectory
            for t, (o, a, r) in enumerate(zip(self.rb.visible(_registers), _actions, _rewards)):
                alg.remember(o, a, r)

            e += 1

        alg.end_episode()

        registers = np.concatenate(list(registers) + [np.expand_dims(final_registers, 0)], axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)

        info.append(dict(external_obs=external_obs))

        return registers, actions, rewards, info

    def visualize(self, policy, n_rollouts=None, T=None, mode='train', render_rollouts=None):
        """ Visualize rollouts. """
        rollout_results = self.do_rollouts(DummyAlg(), policy, n_rollouts, T, mode)
        self._pprint_rollouts(*rollout_results)
        if render_rollouts is not None:
            render_rollouts(self, *rollout_results)

    def _pprint_rollouts(self, registers, actions, rewards, info):
        external_step_lengths = [i['length'] for i in info[:-1]]
        external_obs = [i['external_obs'] for i in info]

        total_internal_steps = sum(external_step_lengths)

        row_names = ['t=', 'i=', ''] + self.internal.action_names

        omit = set(self.rb.no_display)
        reg_ranges = {}

        for n, s in zip(self.rb.names, self.rb.shapes):
            if n in omit:
                continue

            row_names.append('')
            start = len(row_names)
            for k in range(s):
                row_names.append('{}[{}]'.format(n, k))
            end = len(row_names)
            reg_ranges[n] = (start, end)
        row_names.extend(['', 'reward'])

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
                for n, v in zip(self.rb.names, registers):
                    if n in omit:
                        continue
                    rr = reg_ranges[n]
                    values[i, rr[0]:rr[1]] = v[i, b, :]
                values[i, -1] = rewards[i, b]

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
