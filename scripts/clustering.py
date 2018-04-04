import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from kmodes.kmodes import KModes
import imageio
from io import BytesIO
from scipy.spatial.distance import pdist, squareform

from dps.utils import sha_cache, NumpySeed
from dps.datasets.atari import StaticAtariDataset


@sha_cache("cluster_cache")
def f(game, modes, K, N, in_colour, seed):
    print("Running clustering...")
    with NumpySeed(seed):
        dset = StaticAtariDataset(game=game, after_warp=not in_colour)

        X = dset.x

        if N:
            X = X[:N, ...]
        else:
            N = X.shape[0]

        if not in_colour:
            X = X[..., 0]
        image_shape = X.shape[1:]
        X = X.reshape(N, -1)

        if modes:
            km = KModes(n_clusters=K, init='Huang', n_init=1, verbose=1)
            km.fit(X)

            centroids = km.cluster_centroids_
            centroids = centroids.reshape(K, *image_shape)
            discrete_centroids = centroids
            centroids = centroids / 255.

            labels = km.labels_
        else:
            result = k_means(X / 255., K)
            centroids = result[0]
            discrete_centroids = np.uint8(np.floor(centroids * 255))

        centroids = np.maximum(centroids, 1e-6)
        centroids = np.minimum(centroids, 1-1e-6)
        centroids = centroids.reshape(K, *image_shape)

        labels = np.array(labels)
        X = X.reshape(N, *image_shape)
        return centroids, discrete_centroids, labels, X
    print("Done.")


# game = "IceHockeyNoFrameskip-v4"
# K = 10
game = "BankHeistNoFrameskip-v4"
K = 10

N = None
modes = False
in_colour = False
seed = 0

centroids, discrete_centroids, labels, X = f(game, modes, K, N, in_colour, seed)

hamming_distance = squareform(pdist(discrete_centroids.reshape(K, -1), "hamming"))
print(hamming_distance)

M = centroids.reshape(K, -1).shape[1]
print(M * hamming_distance)

n_plots = 6
fig, axes = plt.subplots(K, n_plots)
for i, centroid in enumerate(discrete_centroids):
    with BytesIO() as output:
        imageio.imwrite(output, centroid, format="PNG")
        contents = output.getvalue()

    n_members = (labels == i).sum()

    ax = axes[i, 0]
    ax.set_title("File size: {}, # of members: {}".format(len(contents), n_members))
    ax.imshow(centroid)

    indices = np.nonzero(labels == i)[0][:n_plots-1]

    for j, idx in zip(range(1, n_plots), indices):
        axes[i, j].imshow(X[idx])

plt.show()
