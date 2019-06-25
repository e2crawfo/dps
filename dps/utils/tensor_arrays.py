import tensorflow as tf

"""
Code for working with tensor arrays, especially in a tf.while_loop, and especially complex
nested structures of tensor arrays.
"""


def apply_keys(d, values):
    """ Assumes `d` is a dict-like structured rep, and `values` is a list-like structured
        rep, and that they have the same structure when `d` is sorted by key. Then basically
        creates a dict that takes keys from `d` and values from `values`.

        Like a structured version of:
        return {k: _v for k, _v in zip(sorted(d), values)}

    """
    new = type(d)()
    for (k, v), _v in zip(sorted(d.items()), values):
        if isinstance(_v, tf.Tensor):
            new[k] = _v
        else:
            new[k] = apply_keys(v, _v)
    return new


def append_to_tensor_arrays(f, structured, tensor_arrays):
    new_tensor_arrays = []
    for (k, v), ta in zip(sorted(structured.items()), tensor_arrays):
        if isinstance(v, tf.Tensor):
            result = ta.write(f, v)
        else:
            result = append_to_tensor_arrays(f, v, ta)
        new_tensor_arrays.append(result)
    return new_tensor_arrays


def make_tensor_arrays(structure, n_frames, prefix=""):
    tas = []
    for k, v in sorted(structure.items()):
        name = prefix + "-" + k if prefix else k

        if isinstance(v, tf.Tensor):
            ta = tf.TensorArray(v.dtype, n_frames, dynamic_size=False, element_shape=v.shape, name=name)
            tas.append(ta)
        else:
            _tas = make_tensor_arrays(v, n_frames, prefix=name)
            tas.append(_tas)
    return tas
