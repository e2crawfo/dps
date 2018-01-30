import gym
import numpy as np

from dps.datasets import SupervisedDataset
from dps.utils import Param


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def gather_atari_frames(game, policy, n_frames, density=1.0, render=False):
    assert 0 < density <= 1.0

    env = gym.make(game)
    if policy is None:
        policy = RandomAgent(env.action_space)

    if render:
        outdir = '/tmp/random-agent-results'
        env = gym.wrappers.Monitor(env, directory=outdir, force=True)

    env.seed(0)
    np.random.seed(0)

    reward = 0
    done = False
    frames = []

    while len(frames) < n_frames:
        ob = env.reset()
        while True:
            action = policy.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if np.random.binomial(1, density):
                frames.append(ob)
            if done:
                break
            if render:
                env.render()

    env.close()
    return np.array(frames[:n_frames])


class AtariAutoencodeDataset(SupervisedDataset):
    game = Param(aliases="atari_game")
    policy = Param()

    def __init__(self, **kwargs):
        frames = gather_atari_frames(self.game, self.policy, self.n_examples)
        super(AtariAutoencodeDataset, self).__init__(frames, frames)


if __name__ == "__main__":
    dset = AtariAutoencodeDataset(game="AsteroidsNoFrameskip-v4", policy=None, n_examples=1000)
