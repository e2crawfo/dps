import tensorflow as tf
import numpy as np

from dps import CoreNetwork
from dps.register import RegisterBank
from dps.environment import RegressionEnv
from dps.utils import default_config, MLP
from dps.production_system import ProductionSystemTrainer
from dps.train import build_and_visualize
from dps.policy import Policy
from dps.experiments import mnist
from dps.experiments.translated_mnist import MnistDrawPretrained, render_rollouts


class MnistArithmeticEnv(RegressionEnv):
    def __init__(self, simple, base, n_digits, W, N, n_train, n_val, n_test, inc_delta, inc_x, inc_y):
        self.simple = simple
        self.base = base
        self.n_digits = n_digits
        self.W = W
        self.N = N
        self.inc_delta = inc_delta
        self.inc_x = inc_x
        self.inc_y = inc_y
        max_overlap = 100

        if simple:
            # We only learn about one function, addition, and there is no symbol to extract.
            super(MnistArithmeticEnv, self).__init__(
                train=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_train, for_eval=False),
                val=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_val, for_eval=True),
                test=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_test, for_eval=True))
        else:
            symbols = [('A', lambda x: sum(x)), ('M', lambda x: np.product(x)), ('C', lambda x: len(x))]
            train = mnist.MnistArithmeticDataset(
                symbols, W, n_digits, max_overlap, n_train, base=base, for_eval=False, shuffle=True)
            val = mnist.MnistArithmeticDataset(
                symbols, W, n_digits, max_overlap, n_train, base=base, for_eval=True)
            test = mnist.MnistArithmeticDataset(
                symbols, W, n_digits, max_overlap, n_train, base=base, for_eval=True)
            super(MnistArithmeticEnv, self).__init__(train=train, val=val, test=test)

    def __str__(self):
        return "<MnistArithmeticEnv W={}>".format(self.W)

    def _render(self, mode='human', close=False):
        pass


def build_classifier(inp, outp_size):
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, outp_size)
    return tf.nn.softmax(logits)


class MnistArithmetic(CoreNetwork):
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
        self.inc_delta = env.inc_delta
        self.inc_x = env.inc_x
        self.inc_y = env.inc_y

        build_classifier = default_config().build_classifier
        classifier_str = default_config().classifier_str

        name = '{}_N={}_symbols={}.chk'.format(classifier_str, self.N, '_'.join(str(s) for s in range(10)))
        self.digit_classifier = MnistDrawPretrained(build_classifier, self.N, name=name, var_scope_name='digit_classifier')

        op_symbols = [10, 12, 22]
        name = '{}_N={}_symbols={}.chk'.format(classifier_str, self.N, '_'.join(str(s) for s in op_symbols))
        mnist_config = mnist.MnistConfig(symbols=op_symbols)
        self.op_classifier = MnistDrawPretrained(build_classifier, self.N, name=name, config=mnist_config, var_scope_name='op_classifier')

        values = (
            [0., 0., 0., 0., 1., 0., 0., 0.] +
            [np.zeros(self.N * self.N, dtype='f')] +
            [np.zeros(self.W * self.W, dtype='f')])

        self.register_bank = RegisterBank(
            'MnistArithmeticRB',
            'op acc fovea_x fovea_y delta vision op_vision t glimpse', 'inp', values=values,
            input_names='inp', output_names='acc', no_display='inp glimpse')
        super(MnistArithmetic, self).__init__()

    def __call__(self, action_activations, r):
        _op, _acc, _fovea_x, _fovea_y, _delta, _vision, _op_vision, _t, _glimpse, _inp = self.register_bank.as_tuple(r)

        (inc_fovea_x, dec_fovea_x, inc_fovea_x_big, dec_fovea_x_big,
         inc_fovea_y, dec_fovea_y, inc_fovea_y_big, dec_fovea_y_big,
         inc_delta, dec_delta, inc_delta_big, dec_delta_big,
         store_op, add, inc, multiply, store, no_op) = (
            tf.split(action_activations, self.n_actions, axis=1))

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

        inp = tf.reshape(_inp, (-1, self.W, self.W))

        digit_classification, glimpse = self.digit_classifier.build_pretrained(
            inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta)
        # batch_size = tf.shape(digit_classification)[0]
        # _vision = digit_classification * tf.tile(tf.expand_dims(tf.range(10, dtype=tf.float32), 0), (batch_size, 1))
        # vision = tf.reduce_sum(_vision, 1, keep_dims=True)
        vision = tf.cast(tf.expand_dims(tf.argmax(digit_classification, 1), 1), tf.float32)

        op_classification, _ = self.op_classifier.build_pretrained(
            inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta)
        # batch_size = tf.shape(classification)[0]
        # _op_vision = classification * tf.tile(tf.expand_dims(tf.range(10, dtype=tf.float32), 0), (batch_size, 1))
        # op_vision = tf.reduce_sum(_op_vision, 1, keep_dims=True)
        op_vision = tf.cast(tf.expand_dims(tf.argmax(op_classification, 1), 1), tf.float32)

        t = _t + 1

        with tf.name_scope("MnistArithmetic"):
            new_registers = self.register_bank.wrap(
                inp=tf.identity(inp, "inp"),
                glimpse=tf.reshape(glimpse, (-1, self.N*self.N), name="glimpse"),
                acc=tf.identity(acc, "acc"),
                op=tf.identity(op, "op"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                op_vision=tf.identity(op_vision, "op_vision"),
                delta=tf.identity(delta, "delta"),
                t=tf.identity(t, "t"))

        return new_registers


def visualize(config):
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        _config = default_config()
        W = 100
        base = 10
        simple = False
        N = 14
        n_digits = 2
        env = MnistArithmeticEnv(simple, base, n_digits, W, N, 10, 10, 10, inc_delta=0.1, inc_x=0.1, inc_y=0.1)
        cn = MnistArithmetic(env)

        controller = FixedController(list(range(cn.n_actions)), cn.n_actions)
        # controller = FixedController([4, 2, 5, 6, 7, 0, 4, 3, 5, 6, 7, 0], cn.n_actions)
        # controller = FixedController([8], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.schedule('exploration'), 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="mnist_arithmetic_policy")
        return ProductionSystem(env, cn, policy, False, len(controller))

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 16, False, render_rollouts=render_rollouts)


class MnistArithmeticTrainer(ProductionSystemTrainer):
    def build_env(self):
        config = default_config()
        return MnistArithmeticEnv(
            config.simple, config.base, config.n_digits, config.W, config.N,
            config.n_train, config.n_val, config.n_test,
            config.inc_delta, config.inc_x, config.inc_y)

    def build_core_network(self, env):
        return MnistArithmetic(env)
