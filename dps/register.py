import tensorflow as tf
import numpy as np

from keyword import iskeyword


def concat(values, axis=-1, lib=None):
    if isinstance(values[0], tf.Tensor):
        return tf.concat(values, -1)
    elif isinstance(values[0], np.ndarray):
        return np.concatenate(values, -1)
    else:
        raise Exception()


class RegisterBank(object):
    def __init__(
            self, bank_name, visible_names, hidden_names, values,
            output_names, no_display=None):

        if isinstance(visible_names, str):
            visible_names = visible_names.replace(',', ' ').split()
        if isinstance(hidden_names, str):
            hidden_names = hidden_names.replace(',', ' ').split()
        if isinstance(output_names, str):
            output_names = output_names.replace(',', ' ').split()
        if isinstance(no_display, str):
            no_display = no_display.replace(',', ' ').split()

        no_display = set(no_display or [])

        visible_names = list(visible_names or [])
        hidden_names = list(hidden_names or [])
        names = visible_names + hidden_names

        assert all(o in names for o in output_names)
        assert all(n in names for n in no_display)

        assert len(names) == len(values)
        assert names, "Register bank must contain at least one register."

        bank_name = str(bank_name)
        for n in [bank_name] + names:
            if type(n) is not str:
                raise TypeError('Type names and field names must be strings')
            if not n.isidentifier():
                raise ValueError('Type names and field names must be valid '
                                 'identifiers: %r' % n)
            if iskeyword(n):
                raise ValueError('Type names and field names cannot be a '
                                 'keyword: %r' % n)
        seen = set()
        for n in names:
            if n.startswith('_'):
                raise ValueError('Field names cannot start with an underscore: '
                                 '%r' % n)
            if n in seen:
                raise ValueError('Encountered duplicate field n: %r' % n)
            seen.add(n)

        values = [np.array(v) for v in values]
        for v in values:
            assert v.ndim == 0 or v.ndim == 1
        values = [v.reshape(v.size) for v in values]

        self.bank_name = bank_name
        self.names = names
        self.visible_names = visible_names
        self.hidden_names = hidden_names
        self.values = values
        self.shapes = [v.size for v in values]
        self.output_names = output_names
        self.no_display = no_display

        self.width = sum(v.size for v in values)

        visible_width = None
        self._offsets = {}
        offset = 0
        for i, (name, value) in enumerate(zip(names, values)):
            if name in self.hidden_names and visible_width is None:
                visible_width = offset
            start, end = offset, offset + value.size
            self._offsets[name] = (start, end)
            offset += value.size

        if visible_width is None:
            visible_width = offset

        self.visible_width = visible_width
        self.dtype = tf.float32

    def __str__(self):
        s = ["{}(".format(self.__class__.__name__)]
        s.append("    Registers:")
        for name, value in zip(self.names, self.values):
            s.append("        {}, vis: {}, init: {}".format(
                name, name not in self.hidden_names, value))
        s.append(")")
        return '\n'.join(s)

    def __repr__(self):
        return str(self)

    def as_str(self, array):
        if isinstance(array, tf.Tensor):
            lib = 'tf'
        elif isinstance(array, np.ndarray):
            lib = 'np'
        else:
            raise Exception()

        s = ["{}[{}](".format(self.__class__.__name__, lib)]
        s.append("    Registers:")
        for name, value in zip(self.names, self.values):
            s.append("        {}, vis: {}, init: {}".format(
                name, name not in self.hidden_names, value))
        s.append("    Array shape:")
        s.append("        {}".format(array.shape))
        s.append(")")
        return '\n'.join(s)

    def new_array(self, leading_shape, lib='np'):
        """ Create a new register bank using stored initial values. """
        try:
            leading_shape = tuple(leading_shape)
        except TypeError:
            leading_shape = (leading_shape,)

        bcast_shape = tuple([1] * len(leading_shape)) + (-1,)
        values = []
        for k, v in zip(self.names, self.values):
            if lib == 'tf':
                value = tf.tile(v.reshape(bcast_shape), leading_shape + (1,))
            elif lib == 'np':
                value = np.tile(v.reshape(bcast_shape), leading_shape + (1,))
            else:
                raise Exception("Unknown lib {}.".format(lib))
            values.append(value)

        return concat(values)

    def new_placeholder(self, leading_shape):
        return tf.placeholder(tf.float32, leading_shape + (self.width,))

    def get(self, name, array):
        start, end = self._offsets[name]
        return array[..., start:end]

    def set(self, name, array, other):
        start, end = self._offsets[name]
        array[..., start:end] = other

    def as_tuple(self, array, visible_only=False):
        names = self.visible_names if visible_only else self.names
        return tuple(self.get(name, array) for name in names)

    def as_dict(self, array, visible_only=False):
        names = self.visible_names if visible_only else self.names
        return {name: self.get(name, array) for name in names}

    def get_output(self, array):
        values = [self.get(oname, array) for oname in self.output_names]
        return concat(values)

    def visible(self, array):
        return array[..., :self.visible_width]

    def wrap(self, **kwargs):
        names = kwargs.keys()
        missing = set(self.names) - names
        if missing:
            raise Exception("No value provided for registers {}.".format(missing))
        extra = names - set(self.names)
        if extra:
            raise Exception("Value provided for unknown registers {}.".format(extra))
        values = [kwargs[name] for name in self.names]
        return concat(values)
