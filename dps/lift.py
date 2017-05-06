import numpy as np
import tensorflow as tf


def mat_for_lifted_unary(f, input_space, output_space):
    """ Lift a unary function.

    Parameters
    ----------
    f: function
        The function to lift.

    """
    M = np.zeros((len(output_space), len(input_space)), dtype='f')
    output_map = {o: idx for idx, o in enumerate(output_space)}
    for idx, i in enumerate(input_space):
        o = f(i)
        if o not in output_space:
            raise ValueError("Input value {} mapped to an output value {} that was not in the output space.".format(i, o))
        M[output_map[o], idx] = 1.0
    return M


def mat_for_lifted_binary(f, input_space1, input_space2, output_space):
    M = np.zeros((len(output_space), len(input_space1), len(input_space2)), dtype='f')
    output_map = {o: idx for idx, o in enumerate(output_space)}

    for idx, i in enumerate(input_space1):
        for jdx, j in enumerate(input_space2):
            o = f(i, j)
            if o not in output_space:
                raise ValueError("Input value ({}, {}) mapped to an output value {} that was not in the output space.".format(i, j, o))
            M[output_map[o], idx, jdx] = 1.0

    return M


def lift_unary(f, input_space, output_space, input_tensor):
    input_space = list(input_space)
    output_space = list(output_space)
    M = mat_for_lifted_unary(f, input_space, output_space)
    output_tensor = tf.tensordot(input_tensor, M, [1, 1])
    return output_tensor, M


def lift_binary(f, input_space1, input_space2, output_space, input_tensor1, input_tensor2):
    input_space1 = list(input_space1)
    input_space2 = list(input_space2)
    output_space = list(output_space)

    M = mat_for_lifted_binary(f, input_space1, input_space2, output_space)

    partial = tf.tensordot(input_tensor2, M, [[1], [2]])
    # partial now has dimensions (batch_size, |output_space|, |input_space1|

    output_tensor = tf.matmul(partial, tf.expand_dims(input_tensor1, -1))
    # output_tensor now has dimensions (batch_size, output_space, 1)

    output_tensor = tf.squeeze(output_tensor, axis=[-1])
    return output_tensor, M
