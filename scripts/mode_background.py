import matplotlib.pyplot as plt
import numpy as np
import collections

from dps.utils import sha_cache, NumpySeed
from dps.datasets.atari import StaticAtariDataset


def compute_background(data, mode_threshold):
    assert data.dtype == np.uint8
    mask = np.zeros(data.shape[1:3])
    background = np.zeros(data.shape[1:4])

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            print("Doing {}".format((i, j)))
            channel = [tuple(cell) for cell in data[:, i, j, ...]]
            counts = collections.Counter(channel)
            mode, mode_count = counts.most_common(1)[0]
            if mode_count / data.shape[0] > mode_threshold:
                mask[i, j] = 1
                background[i, j, ...] = mode_count
            else:
                mask[i, j] = 0

    return mask, background


@sha_cache("compute_background")
def f(game, N, in_colour, threshold, seed):
    print("Computing background...")
    with NumpySeed(seed):
        dset = StaticAtariDataset(game=game, after_warp=not in_colour)
        X = dset.x
        if N:
            X = X[:N]
        mask, background = compute_background(X, threshold)
        return mask, background


game = "IceHockeyNoFrameskip-v4"
in_colour = False
N = 1000
threshold = 0.8
seed = 0


mask, background = f(game, N, in_colour, threshold, seed)

if not in_colour:
    background = background[..., 0]

fig, axes = plt.subplots(1, 2)

axes[0].imshow(mask)
axes[0].set_title("Mask")

axes[1].imshow(background)
axes[1].set_title("Background")

plt.show()
