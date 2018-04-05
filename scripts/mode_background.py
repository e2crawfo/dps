import matplotlib.pyplot as plt

from dps.utils import sha_cache, NumpySeed
from dps.datasets.atari import StaticAtariDataset
from dps.env.advanced.yolo_rl import compute_background


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
