import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
import numpy as np

from dps import cfg
from dps.utils import Param
from dps.utils.tf import trainable_variables, ScopedFunction, MLP, FullyConvolutional
from dps.env.advanced import yolo_rl, yolo_air
from dps.datasets import VisualArithmeticDataset
from dps.updater import DifferentiableUpdater
from dps.env.supervised import ClassificationEnv


def get_math_updater(env):
    network = YoloAir_MathNetwork(env)
    return yolo_rl.YoloRL_Updater(env, network)


class SequentialRegressionNetwork(ScopedFunction):
    h_cell = None
    w_cell = None
    b_cell = None

    output_network = None

    def _call(self, _inp, output_size, is_training):
        """ program is the program dictionary from YoloAir_Network """

        if self.h_cell is None:
            self.h_cell = cfg.build_math_cell(scope="regression_h_cell")
            self.w_cell = cfg.build_math_cell(scope="regression_w_cell")
            self.b_cell = cfg.build_math_cell(scope="regression_b_cell")

        edge_state = self.h_cell.zero_state(tf.shape(_inp)[0], tf.float32)

        H, W, B = tuple(int(i) for i in _inp.shape[1:4])
        h_states = np.empty((H, W, B), dtype=np.object)
        w_states = np.empty((H, W, B), dtype=np.object)
        b_states = np.empty((H, W, B), dtype=np.object)

        for h in range(H):
            for w in range(W):
                for b in range(B):
                    h_state = h_states[h-1, w, b] if h > 0 else edge_state
                    w_state = w_states[h, w-1, b] if w > 0 else edge_state
                    b_state = b_states[h, w, b-1] if b > 0 else edge_state

                    inp = _inp[:, h, w, b, :]

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


class ConvolutionalRegressionNetwork(ScopedFunction):
    network = None

    def _call(self, inp, output_size, is_training):
        """ inp is the program dictionary from YoloAir_Network """

        if self.network is None:
            self.network = cfg.build_convolutional_network(scope="regression_network")

        return self.network(inp['attr'], output_size, is_training)


class YoloAir_MathNetwork(yolo_air.YoloAir_Network):
    math_weight = Param()
    largest_digit = Param()

    math_input_network = None
    math_network = None

    def trainable_variables(self, for_opt):
        tvars = super(YoloAir_MathNetwork, self).trainable_variables(for_opt)
        math_network_tvars = trainable_variables(self.math_network.scope, for_opt=for_opt)
        tvars.extend(math_network_tvars)
        math_input_network_tvars = trainable_variables(self.math_input_network.scope, for_opt=for_opt)
        tvars.extend(math_input_network_tvars)
        return tvars

    def build_graph(self, *args, **kwargs):
        result = super(YoloAir_MathNetwork, self).build_graph(*args, **kwargs)

        if self.math_input_network is None:
            self.math_input_network = cfg.build_math_input(scope="math_input_network")

            if "math" in self.fixed_weights:
                self.math_input_network.fix_variables()

        attr = tf.reshape(self.program['attr'], (self.batch_size * self.HWB, self.A))
        math_attr = self.math_input_network(attr, self.A, self.is_training)
        math_attr = tf.reshape(math_attr, (self.batch_size, self.H, self.W, self.B, self.A))
        self._tensors["math_attr"] = math_attr

        # Use raw_obj so that there is no discrepancy between validation and train
        _inp = self._tensors["raw_obj"] * math_attr

        if self.math_network is None:
            self.math_network = cfg.build_math_network(scope="math_network")

            if "math" in self.fixed_weights:
                self.math_network.fix_variables()

        logits = self.math_network(_inp, self.largest_digit + 1, self.is_training)

        if self.math_weight is not None:
            result["recorded_tensors"]["raw_loss_math"] = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self._tensors["targets"],
                    logits=logits
                )
            )

            result["losses"]["math"] = self.math_weight * result["recorded_tensors"]["raw_loss_math"]

        self._tensors["prediction"] = tf.nn.softmax(logits)

        result["recorded_tensors"]["math_accuracy"] = tf.reduce_mean(
            tf.to_float(
                tf.equal(
                    tf.argmax(logits, axis=1),
                    tf.argmax(self._tensors['targets'], axis=1)
                )
            )
        )

        result["recorded_tensors"]["math_1norm"] = tf.reduce_mean(
            tf.to_float(
                tf.abs(tf.argmax(logits, axis=1) - tf.argmax(self._tensors['targets'], axis=1))
            )
        )

        result["recorded_tensors"]["math_correct_prob"] = tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax(logits) * self._tensors['targets'], axis=1)
        )

        return result


class Env(object):
    def __init__(self):
        train = VisualArithmeticDataset(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = VisualArithmeticDataset(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


config = yolo_air.config.copy(
    log_name="yolo_air_math",
    get_updater=get_math_updater,
    build_env=Env,

    patch_shape=(14, 14),

    min_digits=1,
    max_digits=3,
    # max_digits=11,

    min_chars=1,
    max_chars=3,
    # max_chars=11,

    largest_digit=99,
    one_hot=True,
    reductions="sum",
    # reductions="A:sum,M:prod,N:min,X:max,C:len",

    build_math_network=SequentialRegressionNetwork,

    build_math_cell=lambda scope: tf.contrib.rnn.LSTMBlockCell(128),
    build_math_output=lambda scope: MLP([100, 100], scope=scope),
    build_math_input=lambda scope: MLP([100, 100], scope=scope),

    math_weight=5.0,
    train_kl=True,
    train_reconstruction=True,
    postprocessing="",

    curriculum=[
        dict(),
    ],

    image_shape=(48, 48),
)

big_config = config.copy(
    image_shape=(84, 84),
    max_digits=12,
    max_chars=12,
)

load_config = big_config.copy(
    max_steps=10000,
    curriculum=[
        dict(fixed_weights="math", math_weight=0.0, postprocessing="random", tile_shape=(48, 48), train_kl=True, train_reconstruction=True),
        dict(preserve_env=False,),
    ],

    math_weight=1.0,
    train_kl=False,
    train_reconstruction=False,
    fixed_weights="object_encoder object_decoder box obj backbone edge",
)


class ConvEnv(ClassificationEnv):
    def __init__(self):
        train = VisualArithmeticDataset(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = VisualArithmeticDataset(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.obs_shape = train.obs_shape
        self.action_shape = train.largest_digit + 1

        super(ConvEnv, self).__init__(train, val)


class ConvNet(FullyConvolutional):
    def __init__(self):
        layout = [
            dict(filters=64, kernel_size=5, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=2, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=2, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=1, padding="VALID"),
        ]
        super(ConvNet, self).__init__(layout, check_output_shape=False)

    def _call(self, inp, output_size, is_training):
        output = super(ConvNet, self)._call(inp, output_size, is_training)
        output = tf.nn.relu(output)  # FullyConvolutional doesn't apply non-linearity to final layer
        size = np.product([int(i) for i in output.shape[1:]])
        output = tf.reshape(output, (tf.shape(inp)[0], size))
        output = fully_connected(output, 100)
        output = fully_connected(output, output_size, activation_fn=None)
        return output


def get_diff_updater(env):
    model = ConvNet()
    return DifferentiableUpdater(env, model)


convolutional_config = big_config.copy(
    log_name="yolo_air_math_convolutional",
    get_updater=get_diff_updater,
    render_hook=None,
    build_env=ConvEnv,
    stopping_criteria="01_loss,min",
    threshold=0.0,
)
