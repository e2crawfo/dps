from dps.utils import DpsConfig
import numpy as np


distributions = dict(
    n_controller_units=[128],
    batch_size=[32],
    entropy_schedule=[
        'constant {}'.format(x)
        for x in 2.**(-np.arange(2, 6))
    ],
    exploration_schedule=[
        'exp 1.0 100000 0.01',
        'exp 1.0 100000 0.1',
        'exp 10.0 100000 0.01',
        'exp 10.0 100000 0.1',
    ],
    lr_schedule=[
        'constant 0.00025',
        'constant 1e-4',
        'constant 1e-5',
        'constant 1e-6',
        'poly 0.00025 100000 1e-6 1',
        'poly 1e-4 100000 1e-6 1',
        'poly 1e-5 100000 1e-6 1',
        'poly 1e-6 100000 0.0 1',
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
    mnist = 0
    op_loc = (0, 0)
    start_loc = (0, 0)

    power_through = False
    optimizer_spec = 'rmsprop'
    max_steps = 100000
    preserve_policy = True
    start_tensorboard = False
    verbose = 0
    visualize = False

    reward_window = 0.5
    test_time_explore = 0.1
    threshold = 0.05
    patience = 20000

    noise_schedule = None

    display_step = 1000
    eval_step = 100
    checkpoint_step = 0
    use_gpu = 0
    slim = True
    n_val = 500


if __name__ == "__main__":
    from dps.parallel.hyper import build_search

    config = Config()

    path = '/tmp/dps/jobs'
    name = 'simple_arithmetic_wed_after_guys_weekend'
    n = 300
    repeats = 10
    alg = 'reinforce'
    task = 'simple_arithmetic'
    job = build_search(path, name, n, repeats, alg, task, False, distributions, config)
    job.run('map', None, False, False)
    job.run('reduce', None, False, False)
