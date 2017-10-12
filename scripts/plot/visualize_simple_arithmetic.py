from dps.utils import DpsConfig
from dps.vision import LeNet
from dps.run import _run
import tensorflow as tf
import numpy as np


class Config(DpsConfig):
    curriculum = [
        dict(T=6, shape=(2, 2), min_digits=2, max_digits=3),
        dict(T=12, shape=(3, 3), min_digits=2, max_digits=3),
        dict(T=20, shape=(4, 4), min_digits=2, max_digits=3),
        dict(T=30, shape=(5, 5), min_digits=2, max_digits=3),
    ]
    base = 10
    gamma = 0.99
    mnist = 1
    op_loc = (0, 0)
    start_loc = (0, 0)

    power_through = False
    optimizer_spec = 'rmsprop'
    max_steps = 100000
    preserve_policy = True
    start_tensorboard = True
    verbose = 0

    reward_window = 0.5
    test_time_explore = 0.1
    threshold = 0.05
    patience = np.inf

    noise_schedule = None

    display = False
    save_display = False
    verbose = False
    display_step = 1000
    eval_step = 100
    checkpoint_step = 0
    use_gpu = 1
    slim = False
    n_val = 500

    classifier_str = "LeNet_256"

    @staticmethod
    def build_classifier(inp, output_size, is_training=False):
        logits = LeNet(256, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)
        return tf.nn.softmax(logits)

    batch_size = 32
    entropy_schedule = 0.1
    exploration_schedule = "Poly(10, 0.1, 100000)"
    lr_schedule = "0.00025"
    n_controller_units = 128


if __name__ == "__main__":
    config = Config()

    n = 300
    repeats = 10
    alg = 'reinforce'
    task = 'simple_arithmetic'

    _run(alg, task, config)
