import tensorflow as tf
import numpy as np

from dps import cfg
from dps.utils import Param
from dps.utils.tf import trainable_variables, ScopedFunction, MLP, LeNet
from dps.env.advanced import yolo_rl
from dps.datasets import VisualArithmetic


def get_math_updater(env):
    network = YoloRL_MathNetwork(env)
    return yolo_rl.YoloRL_Updater(env, network)


def lifted_addition(inp1, inp2):
    """
    Assume both are based at 0.
    inp1: shape=(batch_size, max1)
    inp2: shape=(batch_size, max2)

    Assumes max1, max2 are known statically.

    result: shape=(batch_size, max_1 + max_2 - 1)

    """
    max1 = inp1.shape[1]
    max2 = inp2.shape[1]

    batch_size = tf.shape(inp1)[0]

    indices = tf.range(max1 + max2 - 1)[None, :] - tf.range(max2)[:, None]
    indices = tf.matrix_band_part(indices, 0, max1-1)

    indices = tf.tile(indices[None, :, :], (batch_size, 1, 1))
    batch_indices = tf.range(batch_size)[:, None, None] * tf.ones_like(indices)
    indices = tf.stack([batch_indices, indices], axis=3)

    _inp1 = tf.gather_nd(inp1, indices)
    _inp1 = tf.matrix_band_part(_inp1, 0, max1-1)

    result = _inp1 * inp2[:, :, None]
    result = tf.reduce_sum(result, axis=1)

    return result


class InterpretableAdditionNetwork(ScopedFunction):
    digit_classifier = None

    def _call(self, inp, output_size, is_training):
        """ inp is the program dictionary from YoloRL_Network """

        if self.digit_classifier is None:
            self.digit_classifier = cfg.build_digit_classifier(scope="digit_classifier")

        attrs = inp['attr']
        H, W, B = tuple(int(i) for i in attrs.shape[1:4])
        running_sum = tf.ones((tf.shape(attrs)[0], 1), dtype=tf.float32)
        for h in range(H):
            for w in range(W):
                for b in range(B):
                    digit_logits = self.digit_classifier(attrs[:, h, w, b, :], 10, is_training)
                    digit = tf.nn.softmax(digit_logits)

                    running_sum = lifted_addition(running_sum, digit)
        return running_sum


class SequentialMathNetwork(ScopedFunction):
    h_cell = None
    w_cell = None
    b_cell = None

    output_network = None

    def _call(self, inp, output_size, is_training):
        """ inp is the program dictionary from YoloRL_Network """

        if self.h_cell is None:
            self.h_cell = cfg.build_math_cell(scope="math_h_cell")
            self.w_cell = cfg.build_math_cell(scope="math_w_cell")
            self.b_cell = cfg.build_math_cell(scope="math_b_cell")

        edge_state = self.h_cell.zero_state(tf.shape(inp['attr'])[0], tf.float32)

        attrs = inp['attr']
        H, W, B = tuple(int(i) for i in attrs.shape[1:4])
        h_states = np.empty((H, W, B), dtype=np.object)
        w_states = np.empty((H, W, B), dtype=np.object)
        b_states = np.empty((H, W, B), dtype=np.object)

        for h in range(H):
            for w in range(W):
                for b in range(B):
                    h_state = h_states[h-1, w, b] if h > 0 else edge_state
                    w_state = w_states[h, w-1, b] if w > 0 else edge_state
                    b_state = b_states[h, w, b-1] if b > 0 else edge_state

                    inp = attrs[:, h, w, b, :]

                    h_inp = tf.concat([inp, w_state.h, b_state.h], axis=1)
                    _, h_states[h, w, b] = self.h_cell(h_inp, h_state)

                    w_inp = tf.concat([inp, h_state.h, b_state.h], axis=1)
                    _, w_states[h, w, b] = self.w_cell(w_inp, w_state)

                    b_inp = tf.concat([inp, h_state.h, w_state.h], axis=1)
                    _, b_states[h, w, b] = self.b_cell(b_inp, b_state)

        if self.output_network is None:
            self.output_network = cfg.build_math_output(scope="math_output")

        final_layer_input = tf.concat(
            [h_states[-1, -1, -1].h,
             w_states[-1, -1, -1].h,
             b_states[-1, -1, -1].h],
            axis=1)

        return self.output_network(final_layer_input, output_size, is_training)


class ConvolutionalMathNetwork(ScopedFunction):
    network = None

    def _call(self, inp, output_size, is_training):
        """ inp is the program dictionary from YoloRL_Network """

        if self.network is None:
            self.network = cfg.build_convolutional_network(scope="math_network")

        return self.network(inp['attr'], output_size, is_training)


class YoloRL_MathNetwork(yolo_rl.YoloRL_Network):
    math_weight = Param()
    largest_digit = Param()

    math_network = None

    def trainable_variables(self, for_opt):
        tvars = super(YoloRL_MathNetwork, self).trainable_variables(for_opt)
        math_network_tvars = trainable_variables(self.math_network.scope, for_opt=for_opt)
        tvars.extend(math_network_tvars)
        return tvars

    def build_graph(self, *args, **kwargs):
        result = super(YoloRL_MathNetwork, self).build_graph(*args, **kwargs)

        if self.math_network is None:
            self.math_network = cfg.build_math_network(scope="math_network")

            if "output" in self.fixed_weights:
                self.math_network.fix_variables()

        logits = self.math_network(self.program, self.largest_digit + 1, self.is_training)

        result["recorded_tensors"]["raw_loss_math"] = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self._tensors["targets"],
                logits=logits
            )
        )

        result["losses"]["math"] = self.math_weight * result["recorded_tensors"]["raw_loss_math"]

        return result


class Env(object):
    def __init__(self):
        train = VisualArithmetic(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = VisualArithmetic(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


config = yolo_rl.config.copy(
    get_updater=get_math_updater,
    build_env=Env,

    patch_shape=(14, 14),

    min_digits=1,
    max_digits=12,

    min_chars=1,
    max_chars=12,

    # reductions="A:sum,M:prod,N:min,X:max,C:len",
    reductions="A:sum",

    one_hot=True,
    largest_digit=100,

    build_math_network=SequentialMathNetwork,
    # build_math_network=LeNet,

    # largest_digit=144,  # 16 * 9
    # build_math_network=InterpretableAdditionNetwork,

    build_math_cell=lambda scope: tf.contrib.rnn.LSTMBlockCell(128),
    build_math_output=lambda scope: MLP([100, 100], scope=scope),
    build_digit_classifier=lambda scope: LeNet(128, scope=scope),

    math_weight=1.0,
    curriculum=[
        # dict(math_weight=0.0, do_train=False),
        # dict(math_weight=1.0, postprocessing=""),
        dict(math_weight=1.0),
        # dict(math_weight=1.0),
    ],
)