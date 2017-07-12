from dps.utils import DpsConfig, MLP
from dps.vision import LeNet
import tensorflow as tf


distributions = dict(
    n_controller_units=[128],
    batch_size=[16, 32, 64, 128, 256],
    entropy_schedule=[
        'constant 0.01',
        'constant 0.1',
        'constant 1.0',
    ],
    exploration_schedule=[
        'exp 1.0 100000 0.01',
        'exp 1.0 100000 0.1',
        'exp 10.0 100000 0.01',
        'exp 10.0 100000 0.1',
    ],
    lr_schedule=[
        'constant 0.00025',
    ],
)


class Config(DpsConfig):
    curriculum = [
        dict(T=10, shape=(2, 2), n_digits=2, upper_bound=True),
        dict(T=15, shape=(3, 3), n_digits=2, upper_bound=True),
        dict(T=25, shape=(4, 4), n_digits=2, upper_bound=True),
        dict(T=30, shape=(5, 5), n_digits=2, upper_bound=True),
        # dict(T=2),
        # dict(T=3),
        # dict(T=4),
        # dict(T=5),
        # dict(T=10),
        # dict(T=10, shape=(2, 2)),
        # dict(T=10, n_digits=2, shape=(2, 2)),
        # dict(T=15, n_digits=2, shape=(2, 2)),
        # dict(T=15, n_digits=2, shape=(3, 2)),
        # dict(T=15, n_digits=2, shape=(3, 3)),
        # dict(T=20, n_digits=2, shape=(3, 3)),
        # dict(T=20, n_digits=2, shape=(4, 4)),
        # dict(T=20, n_digits=2, shape=(4, 4)),
    ]
    base = 10
    gamma = 0.99
    upper_bound = True
    mnist = 1
    op_loc = (0, 0)
    start_loc = (0, 0)

    power_through = False
    optimizer_spec = 'rmsprop'
    max_steps = 100000
    preserve_policy = True
    start_tensorboard = True
    verbose = 0
    visualize = False

    reward_window = 0.5
    test_time_explore = 0.1
    threshold = 0.05
    patience = 10000

    noise_schedule = None

    display_step = 1000
    eval_step = 100
    checkpoint_step = 0
    use_gpu = 0
    slim = False
    n_val = 1000

    # classifier_str = "LeNet_256"

    # @staticmethod
    # def build_classifier(inp, output_size, is_training=False):
    #     logits = LeNet(256, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)
    #     return tf.nn.softmax(logits)

    classifier_str = "MLP_50_50"

    @staticmethod
    def build_classifier(inp, outp_size, is_training=False):
        logits = MLP([50, 50], activation_fn=tf.nn.sigmoid)(inp, outp_size)
        return tf.nn.softmax(logits)


if __name__ == "__main__":
    from dps.parallel.hyper import build_search

    config = Config()

    path = '/tmp/dps/jobs'
    name = 'simple_arithmetic_tues_after_guys_weekend'
    n = 300
    repeats = 10
    alg = 'reinforce'
    task = 'simple_arithmetic'
    job = build_search(path, name, n, repeats, alg, task, False, distributions, config)
    job.run('map', None, False, False)
    job.run('reduce', None, False, False)
