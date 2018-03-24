import copy
import numpy as np
import tensorflow as tf

import gym
from gym.spaces import Discrete, Box

from dps import cfg
from dps.env import Env
from dps.rl import RolloutBatch
from dps.utils import gen_seed, Param


class BatchGymEnv(Env):
    gym_env = Param(help="Either an instance of gym's Env class, or a string specifying an env to create.")
    reward_scale = Param(None)

    def __init__(self, **kwargs):
        if isinstance(self.gym_env, str):
            self.env_string = self.gym_env
            self.gym_env = gym.make(self.env_string)
        else:
            self.env_string = ""

        self._env_copies = []
        self._active_envs = []

        assert isinstance(self.gym_env.observation_space, Box)
        self.obs_shape = self.gym_env.observation_space.shape

        assert isinstance(self.gym_env.action_space, Discrete)  # Only works for discrete action spaces for now.
        self.action_shape = (1,)
        self.n_actions = self.gym_env.action_space.n

        self._obs = np.zeros((0,)+self.obs_shape)
        self._done = np.ones((0, 1)).astype('bool')

    def set_mode(self, mode, batch_size):
        assert mode in 'train val test'.split(), "Unknown mode: {}.".format(mode)
        self._mode = mode
        self._batch_size = batch_size

        n_needed = batch_size - len(self._env_copies)

        if n_needed > 0:
            self._env_copies.extend(
                [copy.deepcopy(self.gym_env) for i in range(n_needed)])
            self._obs = np.concatenate(
                [self._obs, np.zeros((n_needed,)+self.obs_shape)], axis=0)
            self._done = np.concatenate(
                [self._done, np.ones((n_needed, 1))], axis=0)

        self._active_envs = self._env_copies[:batch_size]
        self.obs = self._obs[:batch_size, ...]
        self.done = self._done[:batch_size, ...]

    def reset(self):
        if cfg.reset_env:
            for i, env in enumerate(self._active_envs):
                self.done[i, 0] = True

        for idx, env in enumerate(self._active_envs):
            if self.done[idx, 0]:
                self.obs[idx, ...] = env.reset()
                self.done[idx, 0] = False

        return self.obs.copy()

    def step(self, actions):
        rewards = []
        info = []

        assert len(actions) == self._batch_size

        for idx, (a, env) in enumerate(zip(actions, self._active_envs)):
            a = np.squeeze(np.array(a))
            if self.done[idx, 0]:
                rewards.append(0.0)
                info.append({})
            else:
                o, r, d, i = env.step(a)
                self.obs[idx, ...] = o
                rewards.append(r)
                self.done[idx, 0] = d
                info.append(i)

        rewards = np.array(rewards).reshape(-1, 1)
        if self.reward_scale:
            rewards /= self.reward_scale

        return (
            self.obs.copy(),
            rewards,
            self.done.copy(),
            info)

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

    def visualize(self, render_rollouts=None, **rollout_kwargs):
        if cfg.show_plots or cfg.save_plots:
            self.do_rollouts(render_mode="human", **rollout_kwargs)

    def do_rollouts(
            self, policy, n_rollouts=None, T=None, exploration=None,
            mode='train', render_mode=None, save_utils=False):

        policy.set_mode(mode)

        if save_utils:
            raise Exception("NotImplemented")

        T = T or cfg.T

        self.set_mode(mode, n_rollouts)

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
                log_probs=log_probs, static=dict(exploration=exploration))
            obs = new_obs
            t += 1

            if render_mode is not None:
                self.render(mode=render_mode)

        return rollouts