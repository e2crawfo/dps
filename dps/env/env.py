import abc
import numpy as np
import tensorflow as tf
import copy
from collections import defaultdict

import gym
from gym import Env as GymEnv
from gym.spaces import Discrete, Box

from dps import cfg
from dps.rl import RolloutBatch
from dps.utils import Parameterized, gen_seed, Param


class Env(Parameterized, GymEnv, metaclass=abc.ABCMeta):
    ta_order = "obs action reward done log_prob entropy util policy_state".split()
    info_shapes = None
    n_rollouts = None
    has_differentiable_loss = False

    def __init__(self, **kwargs):
        self._samplers = {}
        super(Env, self).__init__(**kwargs)

    def set_mode(self, mode, n_rollouts):
        """ Called at the beginning of `do_rollouts`, before `reset`. """
        assert mode in 'train val test'.split(), "Unknown mode: {}.".format(mode)
        self._mode = mode
        self._n_rollouts = n_rollouts

    @property
    def completion(self):
        return 0.0

    def build_reset(self):
        return tf.py_func(self.reset_wrapper, [], tf.float32, name="{}.reset".format(self.__class__.__name__))

    def build_step(self, actions):
        types = (tf.float32,) * (3 + len(self.info_shapes))
        return tf.py_func(self.step_wrapper, [actions], types, name="{}.step".format(self.__class__.__name__))

    def reset_wrapper(self):
        """ Need to make sure all return values have type float32. """
        obs = self.reset()
        return obs.astype('f')

    def step_wrapper(self, action):
        """ tf.py_func requires that a LIST of tensors be returned, but gym returns info as a dictionary.
            Also need to make sure all return values have type float32. """
        obs, reward, done, info = self.step(action)
        _info = [info[key] for key in sorted(self.info_shapes)]

        return [a.astype('f') for a in [obs, reward, done, *_info]]

    @abc.abstractmethod
    def reset(self):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def step(self, action):
        raise Exception("NotImplemented")

    @abc.abstractproperty
    def obs_shape(self):
        """ The shape of one observation (doesn't include batch_shape). """
        raise Exception("NotImplemented")

    @abc.abstractproperty
    def action_shape(self):
        """ The shape of one action (doesn't include batch_shape). """
        raise Exception("NotImplemented")

    def maybe_build_placeholders(self):
        if self.n_rollouts is None:
            self.n_rollouts = tf.placeholder(tf.int32, (), name="n_rollouts")
            self.T = tf.placeholder(tf.int32, (), name="T")
            self.mode = tf.placeholder(tf.string, (), name="mode")

            self.is_training = tf.equal(self.mode, 'train')
            self.is_testing = tf.equal(self.mode, 'test')

    def maybe_build_sampler(self, policy):
        sampler = self._samplers.get(id(policy))
        if not sampler:
            self.maybe_build_placeholders()

            with tf.name_scope("sampler_" + policy.display_name):

                if self.info_shapes is None:
                    # Get info shapes by running the env for one step
                    dummy_n_rollouts = 2
                    self.set_mode(mode="train", n_rollouts=dummy_n_rollouts)
                    self.reset()
                    dummy_action = np.zeros((dummy_n_rollouts,) + self.action_shape)
                    _, _, _, info = self.step(dummy_action)
                    self.info_shapes = {k: v.shape[1:] for k, v in info.items()}

                def cond(step, done, *_):
                    return tf.logical_and(
                        tf.logical_or(self.T <= 0, step < self.T),
                        tf.logical_not(tf.reduce_all(done > 0.5))
                    )

                def body(step, done, policy_state, obs, *tas):
                    (log_prob, action, entropy, util), new_policy_state = policy(obs, policy_state)
                    new_obs, reward, new_done, *info = self.build_step(action)
                    new_obs = tf.reshape(new_obs, (self.n_rollouts, *self.obs_shape))
                    reward = tf.reshape(reward, (self.n_rollouts, 1))
                    new_done = tf.reshape(new_done, (self.n_rollouts, 1))

                    for _info, (_, shape) in zip(info, sorted(self.info_shapes.items())):
                        _info = tf.reshape(_info, (self.n_rollouts, *shape))

                    values = [obs, action, reward, done, log_prob, entropy, util, policy_state, *info]

                    new_tas = []
                    for ta, val in zip(tas, values):
                        new_ta = ta.write(ta.size(), tf.to_float(val))
                        new_tas.append(new_ta)

                    return (step+1, new_done, new_policy_state, new_obs, *new_tas)

                done = tf.fill((self.n_rollouts, 1), 0.0)
                policy_state = policy.zero_state(self.n_rollouts, tf.float32)
                obs = self.build_reset()
                obs = tf.reshape(obs, (self.n_rollouts, *self.obs_shape))

                n_tas = len(self.ta_order) + len(self.info_shapes)
                inp = (
                    [0, done, policy_state, obs] +
                    [tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) for i in range(n_tas)]
                )

                # Force build_step to be called in a safe (non-loop) environment for the first time.
                dummy_action = tf.zeros((self.n_rollouts,) + self.action_shape)
                self.build_step(dummy_action)

                _, final_done, final_policy_state, final_obs, *tas = tf.while_loop(
                    cond, body, inp, parallel_iterations=1)

                obs_ta, action_ta, reward_ta, done_ta, log_prob_ta, entropy_ta, util_ta, policy_state_ta, *info_tas = tas

                rollout = dict(
                    obs=tf.concat([obs_ta.stack(), final_obs[None, ...]], axis=0),
                    actions=action_ta.stack(),
                    rewards=reward_ta.stack(),
                    done=tf.concat([done_ta.stack()[1:], final_done[None, :, :]], axis=0),
                    log_probs=log_prob_ta.stack(),
                    entropy=entropy_ta.stack(),
                    utils=util_ta.stack(),
                    policy_states=tf.concat([policy_state_ta.stack(), final_policy_state[None, ...]], axis=0)
                )

                assert len(self.info_shapes) == len(info_tas)
                info = {k: ta.stack() for k, ta in zip(sorted(self.info_shapes), info_tas)}
                intersection = info.keys() & rollout.keys()
                assert not intersection, "Info names cannot overlap with rollout names: {}".format(intersection)

                rollout.update(info)

                static = dict(exploration=policy.exploration)

            self._samplers[id(policy)] = rollout, static
            sampler = self._samplers[id(policy)]

        return sampler

    def do_rollouts(self, policy, n_rollouts=None, T=None, exploration=None, mode='train'):
        # Important to do this first, it can mess with the mode and other things.
        rollout, static = self.maybe_build_sampler(policy)

        policy.set_mode(mode)
        self.set_mode(mode, n_rollouts)

        T = T or cfg.T or 0

        feed_dict = {self.n_rollouts: n_rollouts, self.T: T, self.mode: mode}

        if exploration is not None:
            feed_dict.update({policy.exploration: exploration})

        sess = tf.get_default_session()

        _rollout, _static = sess.run([rollout, static], feed_dict=feed_dict)

        return RolloutBatch(**_rollout, static=_static)

    def do_slow_rollouts(
            self, policy, n_rollouts=None, T=None, exploration=None,
            mode='train', render_mode=None):

        policy.set_mode(mode)

        T = T or cfg.T

        obs = self.reset()
        batch_size = obs.shape[0]

        policy_state = policy.zero_state(batch_size, tf.float32)
        policy_state = tf.get_default_session().run(policy_state)

        rollouts = RolloutBatch()

        t = 0

        done = [False]
        while not all(done):
            if T is not None and t >= T:
                break
            (log_probs, action, entropy, utils), policy_state = policy.act(obs, policy_state, exploration)
            new_obs, reward, done, info = self.step(action)
            rollouts.append(
                obs, action, reward, done=done, entropy=entropy,
                log_probs=log_probs, static=dict(exploration=exploration), info=info)
            obs = new_obs
            t += 1

            if render_mode is not None:
                self.render(mode=render_mode)

        return rollouts

    def visualize(self, render_rollouts=None, **rollout_kwargs):
        if cfg.show_plots or cfg.save_plots:
            self.do_slow_rollouts(render_mode="human", **rollout_kwargs)


class TensorFlowEnv(Env):
    @abc.abstractmethod
    def build_reset(self, registers):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def build_step(self, t, registers, action):
        raise Exception("NotImplemented")


class BatchGymEnv(Env):
    gym_env = Param(help="Either an instance of gym's Env class, or a string specifying an env to create.")
    reward_scale = Param(None)

    def __init__(self, **kwargs):
        super(BatchGymEnv, self).__init__()
        if isinstance(self.gym_env, str):
            self.env_string = self.gym_env
            self.gym_env = gym.make(self.env_string)
        else:
            self.env_string = ""

        self._env_copies = []
        self._active_envs = []

        assert isinstance(self.gym_env.observation_space, Box)

        if isinstance(self.gym_env.action_space, Discrete):
            self._action_shape = (1,)
            self.n_actions = self.gym_env.action_space.n
        else:
            self._action_shape = self.gym_env.action_space.shape

        self._obs = np.zeros((0,)+self.obs_shape)
        self._done = np.ones((0, 1)).astype('bool')

    @property
    def obs_shape(self):
        return self.gym_env.observation_space.shape

    @property
    def action_shape(self):
        return self._action_shape

    def set_mode(self, mode, n_rollouts):
        assert mode in 'train val test'.split(), "Unknown mode: {}.".format(mode)
        self._mode = mode
        self._n_rollouts = n_rollouts

        n_needed = n_rollouts - len(self._env_copies)

        if n_needed > 0:
            self._env_copies.extend(
                [copy.deepcopy(self.gym_env) for i in range(n_needed)])

            self._obs = np.concatenate(
                [self._obs, np.zeros((n_needed,)+self.obs_shape)], axis=0)

            self._done = np.concatenate(
                [self._done, np.ones((n_needed, 1))], axis=0)

        self._active_envs = self._env_copies[:n_rollouts]
        self.obs = self._obs[:n_rollouts, ...]
        self.done = self._done[:n_rollouts, ...]

    def reset(self):
        for i, env in enumerate(self._active_envs):
            self.done[i, 0] = True

        for idx, env in enumerate(self._active_envs):
            if self.done[idx, 0]:
                self.obs[idx, ...] = env.reset()
                self.done[idx, 0] = False

        return self.obs.copy()

    def step(self, actions):
        rewards = []
        info = defaultdict(list)
        assert len(actions) == self._n_rollouts

        for idx, (a, env) in enumerate(zip(actions, self._active_envs)):
            if self.done[idx, 0]:
                rewards.append(0.0)
                info.append({})
            else:
                o, r, d, i = env.step(a)
                self.obs[idx, ...] = o
                rewards.append(r)
                self.done[idx, 0] = d

                for k, v in i.items():
                    info[k].append(v)

        rewards = np.array(rewards).reshape(-1, 1)
        if self.reward_scale:
            rewards /= self.reward_scale

        info = {k: np.array(v) for k, v in info.items()}

        return (self.obs.copy(), rewards, self.done.copy(), info)

    def render(self, mode='human'):
        self._active_envs[0].render(mode=mode)

    def close(self):
        for env in self._env_copies:
            env.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        for env in self._env_copies:
            s = gen_seed()
            env.seed(s)
