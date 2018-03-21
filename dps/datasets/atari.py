import gym
import numpy as np

from dps.datasets import ImageDataset
from dps.utils import Param


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def gather_atari_frames(game, policy, n_frames, image_shape=None, density=1.0, render=False):
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


class AtariAutoencodeDataset(ImageDataset):
    game = Param(aliases="atari_game")
    policy = Param()
    image_shape = Param(None)
    samples_per_frame = Param(
        0, help="If 0, scan over each image, extracting as many non-overlapping "
                "sub-images of shape `image_shape` as possible. Otherwise, from each image "
                "we sample `samples_per_frame` sub-images at random.")
    default_shape = (210, 160)

    def __init__(self, **kwargs):
        frames = gather_atari_frames(self.game, self.policy, self.n_examples)
        frame_shape = frames.shape[1:3]
        channel_dim = frames.shape[3]

        if self.image_shape is not None:
            if self.samples_per_frame <= 0:
                assert (
                    (frame_shape[0] % self.image_shape[0] == 0) and
                    (frame_shape[1] % self.image_shape[1] == 0)), (
                        "Frame shape: {}, image shape: {}".format(frame_shape, self.image_shape))

                H = int(frame_shape[0] / self.image_shape[0])
                W = int(frame_shape[1] / self.image_shape[1])

                slices = np.split(frames, W, axis=2)
                new_shape = (self.n_examples * H, self.image_shape[0], self.image_shape[1], channel_dim)
                slices = [np.reshape(s, new_shape) for s in slices]
                frames = np.concatenate(slices, axis=0)
            else:
                _frames = []
                for frame in frames:
                    for j in range(self.samples_per_frame):
                        top = np.random.randint(0, frame_shape[0]-self.image_shape[0]+1)
                        left = np.random.randint(0, frame_shape[1]-self.image_shape[1]+1)

                        image = frame[top:top+self.image_shape[0], left:left+self.image_shape[1], ...]

                        _frames.append(image)
                frames = _frames

        super(AtariAutoencodeDataset, self).__init__(frames)


def show_frames(frames):
    N = len(frames)
    sqrt_N = int(np.ceil(np.sqrt(N)))
    m = int(np.ceil(N / sqrt_N))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(m, sqrt_N)

    for ax, frame in zip(axes.flatten(), frames):
        ax.imshow(frame)
    plt.show()


if __name__ == "__main__":
    game = "SpaceInvadersNoFrameskip-v4"
    # game = "AsteroidsNoFrameskip-v4"
    dset = AtariAutoencodeDataset(game=game, policy=None, n_examples=100, density=0.3)
    show_frames(dset.x[:10])
    # dset = AtariAutoencodeDataset(
    #     game=game, policy=None, n_examples=100, samples_per_frame=2, image_shape=(50, 50))
    # show_frames(dset.x[:100])
    # dset = AtariAutoencodeDataset(
    #     game=game, policy=None, n_examples=100, samples_per_frame=0, image_shape=(30, 40))
