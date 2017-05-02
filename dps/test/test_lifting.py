import tensorflow as tf
import numpy as np

from dps import lifting


def test_lift_binary():
    input_space1 = range(3)
    input_space2 = range(4)

    input_tensor1 = tf.constant([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5], [0.5, 0, 0.5]], dtype=tf.float32)
    input_tensor2 = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0.5, 0.5, 0, 0]], dtype=tf.float32)

    output_tensor, M = lifting.lift_binary(
        lambda x, y: x + y, input_space1, input_space2, range(6), input_tensor1, input_tensor2)
    output_tensor, M = lifting.lift_binary(
        lambda x, y: x * y, input_space1, input_space2, range(7), input_tensor1, input_tensor2)

    sess = tf.Session()
    outp = sess.run(output_tensor)

    ref_M = np.array([
        [[1., 1., 1., 1.],
         [1., 0., 0., 0.],
         [1., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 1., 0.],
         [0., 1., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 1.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 1., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 1.]]], dtype=np.float32)

    assert (M == ref_M).all()

    ref_outp = np.array([[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  0.5 ,  0.  ,  0.5 ,  0.  ,  0.  ],
           [ 0.75,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ]], dtype=np.float32)
    assert (outp == ref_outp).all()
