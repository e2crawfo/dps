import abc
import numpy as np
import tensorflow as tf
import pandas as pd
from tabulate import tabulate

import gym
from gym import Env as GymEnv
from gym.spaces import prng

from dps import cfg
from dps.rl import RolloutBatch
from dps.utils import Parameterized, image_to_string


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
    action_sizes = None
    action_names = None

    # for every entry {name} in this list, a function called `build_{name}` will be
    # called at startup time, and the returned tensor will be recorded throughout training
    recorded_names = []

    def set_mode(self, mode, batch_size):
        """ Called at the beginning of `do_rollouts`. """
        assert mode in 'train val test'.split(), "Unknown mode: {}.".format(mode)
        self._mode = mode
        self._batch_size = batch_size

    @property
    def completion(self):
        return 0.0

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

        segment_lengths = rollouts._metadata.get('segment_lengths', [actions.shape[0]])
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

        n_timesteps, batch_size, *action_shape = actions.shape

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
        if render_rollouts is not None:
            render_rollouts(self, rollouts)
