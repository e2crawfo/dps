from dps.utils import Config
from dps.datasets import EMNIST_ObjectDetection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

config = Config(
    n_examples=10000, use_dataset_cache=True, max_overlap=100, image_shape=(50, 50),
    sub_image_size=(14, 14), backgrounds="red_x red_circle green_x green_circle blue_x blue_circle",
    backgrounds_resize=True, min_chars=1, max_chars=3, sub_image_shape=(14, 14), colours="white")

with config:
    dset = EMNIST_ObjectDetection()

X = dset.x
X = X.reshape(10000, -1)
X = X / 255
K = 10

result = k_means(X, K)
centroids = result[0]
centroids = np.maximum(centroids, 1e-6)
centroids = np.minimum(centroids, 1-1e-6)
centroids = centroids.reshape(K, 50, 50, 3)
fig, axes = plt.subplots(1, K)
for ax, centroid in zip(axes.flatten(), centroids):
    ax.imshow(centroid)

# indices = np.random.choice(10000, replace=False, size=2000)
# _X = X[indices, ...]
# m = BayesianGaussianMixture(K)
# m.fit(_X)
# 
# centroids = m.means_
# centroids = np.maximum(centroids, 1e-6)
# centroids = np.minimum(centroids, 1-1e-6)
# centroids = centroids.reshape(K, 50, 50, 3)
# 
# fig, axes = plt.subplots(1, K)
# for ax, centroid in zip(axes.flatten(), centroids):
#     ax.imshow(centroid)

plt.show()
