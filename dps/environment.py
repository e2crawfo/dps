import abc
from future.utils import with_metaclass

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
from dps.utils import Parameterized, Param


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
        assert mode in 'train train_eval val'.split(), "Unknown mode: {}.".format(mode)
        self.mode = mode
        self.batch_size = batch_size

    def do_rollouts(
            self, policy, n_rollouts=None, T=None, exploration=None,
            mode='train', render_mode=None, save_utils=False):

        T = T or cfg.T

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
        if render_rollouts is not None and (cfg.save_display or cfg.display):
            render_rollouts(self, rollouts)


class RegressionDataset(Parameterized):
    n_examples = Param()

    def __init__(self, x, y, shuffle=True, **kwargs):
        self.x = x
        self.y = y
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0

        super(RegressionDataset, self).__init__(**kwargs)

    @property
    def obs_shape(self):
        return self.x.shape[1:]

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

    loss_type = Param("2-norm")

    def __init__(self, train, val, **kwargs):
        self.train, self.val = train, val
        self.datasets = {
            'train': self.train,
            'train_eval': self.train,
            'val': self.val,
        }

        self.actions_dim = self.train.y.shape[1]
        self.obs_shape = self.train.x.shape[1:]

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
        if self.loss_type == "2-norm":
            return tf.reduce_mean((actions - targets)**2, axis=-1, keep_dims=True)
        elif self.loss_type == "1-norm":
            return tf.reduce_mean(tf.abs(actions - targets), axis=-1, keep_dims=True)
        elif self.loss_type == "xent":
            return tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=actions)

    def build_reward(self, actions, targets):
        if self.loss_type == "xent":
            action_argmax = tf.argmax(actions, axis=-1)
            targets_argmax = tf.argmax(targets, axis=-1)
            reward = tf.cast(tf.equal(action_argmax, targets_argmax), tf.float32) - 1
            return tf.reshape(reward, (-1, 1))
        else:
            abs_error = tf.reduce_sum(tf.abs(actions - targets), axis=-1, keep_dims=True)
            return -tf.cast(abs_error > cfg.reward_window, tf.float32)

    @property
    def completion(self):
        return self.train.completion

    def _step(self, action):
        self.t += 1

        assert self.y.shape == action.shape
        obs = np.zeros(self.x.shape)

        if self.action_ph is None:
            self.target_ph = tf.placeholder(tf.float32, (None, self.actions_dim))
            self.action_ph = tf.placeholder(tf.float32, (None, self.actions_dim))
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
            n_rollouts_ph = tf.placeholder(tf.int32, ())
            T_ph = tf.placeholder(tf.int32, ())

            initial_policy_state = policy.zero_state(n_rollouts_ph, tf.float32)
            initial_registers = self.rb.new_array(n_rollouts_ph, lib='tf')

            sampler_cell = SamplerCell(self, policy)
            with tf.name_scope("sampler"):
                initial_registers = self.build_init(initial_registers)
                t = timestep_tensor(n_rollouts_ph, T_ph)

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

            self._samplers[id(policy)] = (
                n_rollouts_ph, T_ph, initial_policy_state,
                initial_registers, output)

            sampler = self._samplers[id(policy)]

        return sampler

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train', save_utils=False):
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
    dense_reward = Param()
    target_shape = (1,)
    input_shape = None

    def start_episode(self, external_obs, external):
        return (
            external_obs.shape[0],
            {self.input_ph: external_obs,
             self.target_ph: external.y,
             self.is_training_ph: self.mode == 'train'}
        )

    def build_placeholders(self, r):
        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.input_ph = tf.placeholder(tf.float32, (None,) + self.input_shape)
        self.target_ph = tf.placeholder(tf.float32, (None,) + self.target_shape)

    def build_rewards(self, r):
        if self.dense_reward:
            output = self.rb.get_output(r)
            abs_error = tf.reduce_sum(
                tf.abs(output - self.target_ph),
                axis=-1, keep_dims=True)
            rewards = -tf.cast(abs_error > cfg.reward_window, tf.float32)
        else:
            rewards = tf.fill((tf.shape(r)[0], 1), 0.0),
        return rewards


class CompositeEnv(Env):
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
            new_external_obs, external_reward, external_done, i = \
                self.external.step(external_action)

            if not hasattr(self.internal, 'dense_reward') or not self.internal.dense_reward:
                rewards[-1, :, :] += external_reward

            if save_utils:
                rb = RolloutBatch(
                    obs, actions, rewards,
                    log_probs=log_probs, utils=utils,
                    entropy=entropy, hidden=hidden, done=done)
            else:
                rb = RolloutBatch(
                    obs, actions, rewards, log_probs=log_probs,
                    entropy=entropy, hidden=hidden, done=done)
            rollouts.extend(rb)

            external_rollouts.append(
                external_obs, external_action, external_reward, done=external_done,
                info=dict(length=len(rollouts)))

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

            if external_obs[0].shape[-1] < 40:
                print("External observations: ")
                for i, e in enumerate(external_obs):
                    print("{}: {}".format(i, e[b, :]))

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
