import copy
import numpy as np

from gym.spaces import Discrete, Box

from dps.environment import Env
from dps.utils import gen_seed


class GymEnvWrapper(Env):
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self._env_copies = []
        self._active_envs = []
        self.batch_size = None

        assert isinstance(gym_env.observation_space, Box)
        self.obs_shape = gym_env.observation_space.shape

        assert isinstance(gym_env.action_space, Discrete)
        self.n_actions = gym_env.action_space.n

    def set_mode(self, mode, batch_size):
        assert mode in 'train train_eval val'.split(), "Unknown mode: {}.".format(mode)
        self.mode = mode
        self.batch_size = batch_size

        n_needed = batch_size - len(self._env_copies)

        if n_needed > 0:
            self._env_copies.extend(
                [copy.deepcopy(self.gym_env) for i in range(n_needed)])
        self._active_envs = self._env_copies[:batch_size]

    def reset(self):
        obs = []
        for env in self._active_envs:
            obs.append(env.reset())
        self.done = [False for env in self._active_envs]
        self.obs = np.array(obs)
        return self.obs.copy()

    def step(self, actions):
        rewards = []
        info = []

        assert len(actions) == self.batch_size

        for idx, (a, env) in enumerate(zip(actions, self._active_envs)):
            a = np.squeeze(np.array(a))
            if self.done[idx]:
                rewards.append(0.0)
                info.append({})
            else:
                o, r, d, i = env.step(a)
                self.obs[idx, ...] = o
                rewards.append(r)
                self.done[idx] = d
                info.append(i)

        return self.obs.copy(), np.array(rewards).reshape(-1, 1), all(self.done), info

    def render(self, mode='human', close=False):
        self._active_envs[0].render(mode=mode, close=close)

    def close(self):
        for env in self._env_copies:
            env.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        for env in self._env_copies:
            s = gen_seed()
            env.seed(s)

    def completion(self):
        return 0.0

    def visualize(self, render_rollouts=None, **rollout_kwargs):
        self.do_rollouts(render_mode="human", **rollout_kwargs)
