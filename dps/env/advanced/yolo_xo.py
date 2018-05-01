import tensorflow as tf

from dps import cfg
from dps.utils import Param
from dps.utils.tf import trainable_variables, MLP, LeNet, tf_mean_sum
from dps.env.advanced import yolo_rl, yolo_math
from dps.datasets import VisualArithmetic


def get_regression_updater(env):
    network = YoloRL_RegressionNetwork(env)
    return yolo_rl.YoloRL_Updater(env, network)


class YoloRL_RegressionNetwork(yolo_rl.YoloRL_Network):
    regression_weight = Param()
    regression_dim = Param(1)

    regression_network = None

    def trainable_variables(self, for_opt):
        tvars = super(YoloRL_RegressionNetwork, self).trainable_variables(for_opt)
        regression_network_tvars = trainable_variables(self.regression_network.scope, for_opt=for_opt)
        tvars.extend(regression_network_tvars)
        return tvars

    def build_graph(self, *args, **kwargs):
        result = super(YoloRL_RegressionNetwork, self).build_graph(*args, **kwargs)

        if self.regression_network is None:
            self.regression_network = cfg.build_regression_network(scope="regression_network")

            if "regression" in self.fixed_weights:
                self.regression_network.fix_variables()

        output = self.regression_network(self.program, self.regression_dim, self.is_training)

        result["recorded_tensors"]["raw_loss_regression"] = tf_mean_sum((self._tensors["targets"] - output)**2)

        result["losses"]["regression"] = self.regression_weight * result["recorded_tensors"]["raw_loss_regression"]

        return result


class Env(object):
    def __init__(self):
        train = VisualArithmetic(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = VisualArithmetic(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


config = yolo_rl.config.copy(
    get_updater=get_regression_updater,
    build_env=Env,

    build_regression_network=yolo_math.SequentialRegressionNetwork,

    build_regression_cell=lambda scope: tf.contrib.rnn.LSTMBlockCell(128),
    build_regression_output=lambda scope: MLP([100, 100], scope=scope),
    build_digit_classifier=lambda scope: LeNet(128, scope=scope),

    regression_weight=1.0,
    curriculum=[
        # dict(regression_weight=None, do_train=False),
        # dict(regression_weight=1.0, postprocessing=""),
        dict(regression_weight=1.0),
        # dict(regression_weight=1.0),
    ],
)
