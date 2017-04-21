import tensorflow as tf
import numpy as np


class RegisterSpec(object):
    """ Specification of a set of registers.

    Inherit from this class in order to create specific register specifications.

    Parameters
    ----------
    registers: list of union(ndarray, tuple(shape, function))
        Specification of the shape and initial values of the registers.  Each entry
        is either an ndarray or (s, f), where s is a the shape for that register
        and f if a function that accepts a shape and a random state and returns an ndarray.
    visible: list of bool
        If True, corresponding register is visible to the controller.

    """
    dtype = tf.float32

    _visible = None
    _initial_values = None
    _namedtuple = None
    _input_names = None
    _output_names = None

    def assert_defined(self, attr):
        assert getattr(self, attr) is not None, (
            "Instances of subclasses of RegisterSpec must "
            "specify a value for attr {}.".format(attr))

    @property
    def visible(self):
        self.assert_defined('_visible')
        return self._visible

    @property
    def initial_values(self):
        self.assert_defined('_initial_values')
        return self._initial_values

    @property
    def namedtuple(self):
        self.assert_defined('_namedtuple')
        return self._namedtuple

    @property
    def input_names(self):
        self.assert_defined('_input_names')
        return self._input_names

    @property
    def output_names(self):
        self.assert_defined('_output_names')
        return self._output_names

    @property
    def names(self):
        return self.namedtuple._fields

    def get_initial_values(self, batch_size=1, **kwargs):
        """ Returns a list of ndarray's giving initial values for each of the registers. """
        init = []
        for name, t in zip(self.names, self.initial_values):
            if name in kwargs:
                _t = kwargs[name].copy()
                init_value = np.tile(np.expand_dims(_t, 0), (batch_size,) + tuple(np.ones(t.ndim, dtype='i')))
            elif isinstance(t, np.ndarray):
                init_value = np.tile(np.expand_dims(t, 0), (batch_size,) + tuple(np.ones(t.ndim, dtype='i')))
            else:
                shape = t[0]
                f = t[1]
                ivs = [f(shape) for i in batch_size]
                init_value = np.stack(ivs, axis=0)
            init.append(init_value)
        return init

    def shapes(self, visible_only=False):
        return [
            (None,) + (t.shape if isinstance(t, np.ndarray) else t[0])
            for t, v in zip(self.initial_values, self.visible)
            if v or not visible_only]

    def instantiate(self, batch_size=1, **kwargs):
        return self.wrap(*self.get_initial_values(batch_size=batch_size))

    def wrap(self, *args, **kwargs):
        return self.namedtuple(*args, **kwargs)

    def build_placeholders(self, dtype=tf.float32):
        """ Get a list of tensorflow placeholders suitable for feeding as input to this RegisterSpec. """
        shapes = self.shapes()
        return self.wrap(*[tf.placeholder(dtype, shape=s, name=n) for n, s in zip(self.names, shapes)])

    def set_input(self, registers, inp, from_obs=True):
        self.set_register_values(registers, inp, *self.input_names, from_obs=from_obs)

    def get_output(self, registers, as_obs=True):
        return self.get_register_values(registers, *self.output_names, as_obs=as_obs)

    def set_register_values(self, registers, obs, *names, from_obs=True):
        """ If from_obs is True, then ``obs`` has already been split up. """
        values_to_fill = []
        names = names or self.names
        for name in names:
            try:
                vtf = getattr(registers, name)
            except AttributeError:
                try:
                    vtf = registers[name]
                except (TypeError, KeyError):
                    try:
                        vtf = registers[self.names.index(name)]
                        if isinstance(vtf, tuple):
                            vtf = vtf[0]
                    except (TypeError, IndexError):
                        raise Exception(
                            "{} could not be interpreted as an instance of {}, "
                            "it is missing a key/attribute named {}.".format(
                                registers, self, name))
            values_to_fill.append(vtf)

        if not from_obs:
            for vtf, o in zip(values_to_fill, obs):
                vtf[:] = o
        else:
            split_locs = np.cumsum([vtf.shape[-1] for vtf in values_to_fill])
            if isinstance(obs, np.ndarray):
                values = np.split(obs, split_locs[:-1], axis=-1)
            elif isinstance(obs, tf.Tensor):
                values = tf.split(obs, split_locs[:-1], axis=-1)
            else:
                raise Exception(
                    "``obs`` is not recognized datatype, should be either ndarray or Tensor but got: {}.".format(obs))
            for vtf, v in zip(values_to_fill, values):
                vtf[:] = v

    def get_register_values(self, registers, *names, as_obs=True):
        values = []
        names = names or self.names
        for name in names:
            try:
                value = getattr(registers, name)
            except AttributeError:
                try:
                    value = registers[name]
                except (TypeError, KeyError):
                    try:
                        value = registers[self.names.index(name)]
                        if isinstance(value, tuple):
                            value = value[0]
                    except (TypeError, IndexError):
                        raise Exception(
                            "{} could not be interpreted as an instance of {}, "
                            "it is missing a key/attribute named {}.".format(
                                registers, self, name))
            values.append(value)

        if as_obs:
            if len(values) > 1:
                if isinstance(values[0], np.ndarray):
                    obs = np.concatenate(values, axis=-1)
                elif isinstance(values[0], (tf.Tensor, tf.TensorArray)):
                    obs = tf.concat(values, axis=-1)
                else:
                    raise Exception("Register value 0 is not recognized datatype: {}.".format(values[0]))
            else:
                obs = values[0]
        else:
            obs = tuple(values)
        return obs

    def as_obs(self, registers, visible_only=False):
        """ Concatenate values of registers, giving a single ndarray or Tensor representing the entire register.

        Parameters
        ----------
        registers: instance of self.namedtuple
            The object to interpret. For every register in self,
            registers must have an attribute or key with the same name.
        visible_only: bool
            If True, will only include visible register in the returned array.

        """
        if visible_only:
            names = [n for n, v in zip(self.names, self.visible) if v]
        else:
            names = self.names
        return self.get_register_values(registers, *names, as_obs=1)

    def from_obs(self, obs):
        """ Unpack on observation as a register.

        Parameters
        ----------
        obs: ndarray or Tensor
            The observation to unpack. Must have enough dimensions to fill
            all the registers.
        tf: bool
            Whether to use tensorflow or numpy.

        Returns
        -------
        instance of self.namedtuple

        """
        split_locs = np.cumsum([shape[-1] for shape in self.shapes()])
        if isinstance(obs, np.ndarray):
            values = np.split(obs, split_locs[:-1], axis=-1)
        elif isinstance(obs, tf.Tensor):
            values = tf.split(obs, split_locs[:-1], axis=-1)
        else:
            raise Exception("``obs`` is not recognized datatype, should be either ndarray or Tensor but got: {}.".format(obs))
        return self.wrap(*values)

    def state_size(self):
        """ Appropriate for use as the return value of ``RNNCell.state_size()``. """
        ss = []
        for iv in self.initial_values:
            if isinstance(iv, np.ndarray):
                shape = iv.shape
            else:
                shape = iv[0]
            ss.append(shape)
        return ss

    def concatenate(self, reg_list, axis):
        new_registers = []
        for i in range(len(reg_list[0])):
            new_elem = np.concatenate(tuple(r[i][0] if isinstance(r[i], tuple) else r[i] for r in reg_list), axis=axis)
            new_registers.append(new_elem)
        return self.wrap(*new_registers)
