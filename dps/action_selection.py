import tensorflow as tf


def relu_selection(utilities, temperature):
    return tf.nn.relu(utilities, name="relu_action_selection")


def softmax_selection(utilities, temperature):
    aa = tf.nn.softmax(utilities/temperature, name="softmax_action_selection")
    return aa


class gumbel_softmax_selection(object):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Adapted from code by Eric Jang.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes

    """
    def __init__(self, hard=False):
        self.hard = hard

    def __call__(self, utilities, temperature):
        y = gumbel_softmax_sample(utilities, temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature, name='gumbel_softmax')
