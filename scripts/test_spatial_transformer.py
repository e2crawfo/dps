import tensorflow as tf
import numpy as np

import sonnet as snt

from dps.datasets import EMNIST_ObjectDetection

n_examples = 10
image_shape = (28, 28, 3)
crop_shape = (14, 14, 3)

_train = EMNIST_ObjectDetection(
    n_examples=n_examples,
    min_chars=2,
    max_chars=2,
    sub_image_shape=crop_shape[:2],
    characters=[0],
    image_shape=image_shape[:2]).next_batch(n_examples, False)[0]

"""
A = [a, b, tx],
    [c, d, ty]
"""

boxes = np.array([[.5, 0, .5, 0, .5, -.5]], dtype='f')
A = boxes.reshape(2, 3)

# Top left, top right, bottom left bottom right
corners = np.array([[-1, -1, 1], [1, 1, 1]], dtype='f').T
corners = A @ corners
image_A = np.array([[image_shape[1]/2, 0, image_shape[1]/2], [0, image_shape[0]/2, image_shape[0]/2]], dtype='f')  # Transform from grid coords to image coords
image_corners = image_A @ np.concatenate([corners, np.ones((1, corners.shape[1]))], axis=0)

left = image_corners[0, 0]
top = image_corners[1, 0]
right = image_corners[0, 1]
bottom = image_corners[1, 1]
width = right - left
height = bottom - top

boxes = boxes[:, [0, 2, 4, 5]]
boxes = np.tile(boxes, (n_examples, 1))
boxes = tf.constant(boxes, tf.float32)

transform_constraints = snt.AffineWarpConstraints.no_shear_2d()

warper = snt.AffineGridWarper(image_shape[:2], crop_shape[:2], transform_constraints)
grid_coords = warper(boxes)
output = tf.contrib.resampler.resampler(_train, grid_coords)

sess = tf.Session()
crops, _grid_coords = sess.run([output, grid_coords])
print(_grid_coords)

# import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(n_examples, 2)

for image, crop, ax in zip(_train, crops, axes):
    ax[0].imshow(image)

    rect = patches.Rectangle(
        (left, top), width, height, linewidth=1,
        edgecolor="red", facecolor='none')

    ax[0].add_patch(rect)

    ax[1].imshow(crop)

plt.savefig('test_spatial.pdf')