import tensorflow as tf
import numpy as np

from dps.register import RegisterBank


def test_dynamic_partition():
    DummyRB = RegisterBank(
        'DummyRB',
        'r0 r1 r2', None,
        [[1.0, 1.0], 0.0, [0.0, 10]],
        'r0',
        'r0 r1')

    batch_size = 10
    n_actions = 3

    data = np.random.random((batch_size, DummyRB.width))
    actions = np.random.randint(n_actions, size=batch_size)

    actions_ph = tf.placeholder(tf.int32, batch_size)
    partitions = tf.dynamic_partition(data, actions, n_actions)

    sess = tf.Session()

    for i in range(n_actions):
        _r0, _r1, _r2 = DummyRB.as_tuple(partitions[i])
        r0, r1, r2 = sess.run([_r0, _r1, _r2], feed_dict={actions_ph: actions})

        assert np.allclose(r0, data[actions == i, 0:2])
        assert np.allclose(r1, data[actions == i, 2:3])
        assert np.allclose(r2, data[actions == i, 3:5])


if __name__ == "__main__":
    test_dynamic_partition()
