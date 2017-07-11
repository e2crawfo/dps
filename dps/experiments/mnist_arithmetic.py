import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import (
    RegressionEnv, CompositeEnv, TensorFlowEnv)
from dps.vision import (
    TranslatedMnistDataset, MnistArithmeticDataset, DRAW,
    MnistPretrained, MNIST_CONFIG, ClassifierFunc)


class MnistArithmeticEnv(RegressionEnv):
    def __init__(self, simple, base, n_digits, upper_bound, W, N, n_train, n_val, n_test, inc_delta, inc_x, inc_y):
        self.simple = simple
        self.base = base
        self.n_digits = n_digits
        self.upper_bound = upper_bound
        self.W = W
        self.N = N
        self.inc_delta = inc_delta
        self.inc_x = inc_x
        self.inc_y = inc_y
        max_overlap = 100

        if simple:
            kwargs = dict(W=W, n_digits=n_digits, max_overlap=max_overlap)
            super(MnistArithmeticEnv, self).__init__(
                train=TranslatedMnistDataset(n_examples=n_train, for_eval=False, **kwargs),
                val=TranslatedMnistDataset(n_examples=n_val, for_eval=True, **kwargs),
                test=TranslatedMnistDataset(n_examples=n_test, for_eval=True, **kwargs))
        else:
            reductions = [('A', lambda x: sum(x)), ('M', lambda x: np.product(x)), ('C', lambda x: len(x))]
            kwargs = dict(
                W=W, n_digits=n_digits, max_overlap=max_overlap,
                reductions=reductions, upper_bound=upper_bound, base=base)

            train = MnistArithmeticDataset(
                n_examples=n_train, for_eval=False, shuffle=True, **kwargs)
            val = MnistArithmeticDataset(
                n_examples=n_train, for_eval=True, shuffle=False, **kwargs)
            test = MnistArithmeticDataset(
                n_examples=n_train, for_eval=True, shuffle=False, **kwargs)

        super(MnistArithmeticEnv, self).__init__(train=train, val=val, test=test)

    def __str__(self):
        return "<MnistArithmeticEnv W={}>".format(self.W)

    def _render(self, mode='human', close=False):
        pass


class MnistArithmetic(TensorFlowEnv):
    """ Top left is (y=0, x=0). Corresponds to using origin='upper' in plt.imshow.

    Need 2 working memories: one for operation, one for accumulator.

    Assume the operations accept the output of vision and the accumulator as their input,
    and store the result in the accumulator. Having the operations be more general is
    in terms of their input arguments is a matter of neural engineering.

    """
    action_names = [
        'fovea_x += ', 'fovea_x -= ', 'fovea_x ++= ', 'fovea_x --= ',
        'fovea_y += ', 'fovea_y -= ', 'fovea_y ++= ', 'fovea_y --= ',
        'delta += ', 'delta -= ', 'delta ++= ', 'delta --= ',
        'store_op', 'add', 'inc', 'multiply', 'store', 'no-op/stop']

    def __init__(self, env):
        self.W = env.W
        self.N = env.N
        self.upper_bound = env.upper_bound
        self.inc_delta = env.inc_delta
        self.inc_x = env.inc_x
        self.inc_y = env.inc_y
        self.base = env.base

        self.build_attention = DRAW(self.N)

        digit_config = MNIST_CONFIG.copy(symbols=range(self.base))

        name = '{}_N={}_symbols={}.chk'.format(cfg.classifier_str, self.N, '_'.join(str(s) for s in range(self.base)))
        digit_pretrained = MnistPretrained(
            self.build_attention, cfg.build_classifier, name=name,
            var_scope_name='digit_classifier', mnist_config=digit_config)
        self.build_digit_classifier = ClassifierFunc(digit_pretrained, self.base + 1)

        op_config = MNIST_CONFIG.copy(symbols=[10, 12, 22])

        name = '{}_N={}_symbols={}.chk'.format(cfg.classifier_str, self.N, '_'.join(str(s) for s in op_config.symbols))
        op_pretrained = MnistPretrained(
            self.build_attention, cfg.build_classifier, name=name,
            var_scope_name='op_classifier', mnist_config=op_config)
        self.build_op_classifier = ClassifierFunc(op_pretrained, len(op_config.symbols) + 1)

        values = (
            [0., 0., 0., 0., 1., 0., 0.] +
            [np.zeros(self.N * self.N, dtype='f')])

        self.rb = RegisterBank(
            'MnistArithmeticRB',
            'op acc fovea_x fovea_y delta vision op_vision glimpse', None, values=values,
            output_names='acc', no_display='glimpse')
        super(MnistArithmetic, self).__init__()

    def static_inp_type_and_shape(self):
        return (tf.float32, (self.W*self.W,))

    make_input_available = True

    def build_init(self, r, inp):
        op, acc, fovea_x, fovea_y, delta, vision, op_vision, glimpse = self.rb.as_tuple(r)

        glimpse = self.build_attention(inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, sigma=1.0)

        digit_classification = tf.stop_gradient(self.build_digit_classifier(glimpse))
        vision = tf.cast(tf.expand_dims(tf.argmax(digit_classification, 1), 1), tf.float32)

        op_classification = tf.stop_gradient(self.build_op_classifier(glimpse))
        op_vision = tf.cast(tf.expand_dims(tf.argmax(op_classification, 1), 1), tf.float32)

        with tf.name_scope("MnistArithmetic"):
            new_registers = self.rb.wrap(
                glimpse=tf.reshape(glimpse, (-1, self.N*self.N), name="glimpse"),
                acc=tf.identity(acc, "acc"),
                op=tf.identity(op, "op"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                op_vision=tf.identity(op_vision, "op_vision"),
                delta=tf.identity(delta, "delta"))

        return new_registers

    def build_step(self, t, r, a, inp):
        _op, _acc, _fovea_x, _fovea_y, _delta, _vision, _op_vision, _glimpse = self.rb.as_tuple(r)

        (inc_fovea_x, dec_fovea_x, inc_fovea_x_big, dec_fovea_x_big,
         inc_fovea_y, dec_fovea_y, inc_fovea_y_big, dec_fovea_y_big,
         inc_delta, dec_delta, inc_delta_big, dec_delta_big,
         store_op, add, inc, multiply, store, no_op) = (
            tf.split(a, self.n_actions, axis=1))

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (_vision + _acc) + \
            multiply * (_vision * _acc) + \
            inc * (_acc + 1) + \
            store * _vision
        op = (1 - store_op) * _op + store_op * _op_vision

        fovea_x = (1 - inc_fovea_x - dec_fovea_x - inc_fovea_x_big - dec_fovea_x_big) * _fovea_x + \
            inc_fovea_x * (_fovea_x + self.inc_x) + \
            inc_fovea_x_big * (_fovea_x + 5 * self.inc_x) + \
            dec_fovea_x * (_fovea_x - self.inc_x) + \
            dec_fovea_x_big * (_fovea_x - 5 * self.inc_x)

        fovea_y = (1 - inc_fovea_y - dec_fovea_y - inc_fovea_y_big - dec_fovea_y_big) * _fovea_y + \
            inc_fovea_y * (_fovea_y + self.inc_y) + \
            inc_fovea_y_big * (_fovea_y + 5 * self.inc_y) + \
            dec_fovea_y * (_fovea_y - self.inc_y) + \
            dec_fovea_y_big * (_fovea_y - 5 * self.inc_y)

        delta = (1 - inc_delta - dec_delta - inc_delta_big - dec_delta_big) * _delta + \
            inc_delta * (_delta + self.inc_delta) + \
            inc_delta_big * (_delta + 5 * self.inc_delta) + \
            dec_delta * (_delta - self.inc_delta) + \
            dec_delta_big * (_delta - 5 * self.inc_delta)

        glimpse = self.build_attention(inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, sigma=1.0)

        digit_classification = tf.stop_gradient(self.build_digit_classifier(glimpse))
        vision = tf.cast(tf.expand_dims(tf.argmax(digit_classification, 1), 1), tf.float32)

        op_classification = tf.stop_gradient(self.build_op_classifier(glimpse))
        op_vision = tf.cast(tf.expand_dims(tf.argmax(op_classification, 1), 1), tf.float32)

        with tf.name_scope("MnistArithmetic"):
            new_registers = self.rb.wrap(
                glimpse=tf.reshape(glimpse, (-1, self.N*self.N), name="glimpse"),
                acc=tf.identity(acc, "acc"),
                op=tf.identity(op, "op"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                op_vision=tf.identity(op_vision, "op_vision"),
                delta=tf.identity(delta, "delta"))

        return tf.fill((tf.shape(r)[0], 1), 0.0), new_registers


def build_env():
    external = MnistArithmeticEnv(
        cfg.simple, cfg.base, cfg.n_digits, cfg.upper_bound, cfg.W, cfg.N,
        cfg.n_train, cfg.n_val, cfg.n_test,
        cfg.inc_delta, cfg.inc_x, cfg.inc_y)
    internal = MnistArithmetic(external)
    return CompositeEnv(external, internal)
