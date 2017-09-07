import numpy as np

from dps.utils import Config
from dps.rl import QLearning
from dps.config import get_updater, tasks, DuelingLstmController, epsilon_greedy


config = tasks['alt_arithmetic']


config.update(Config(
    curriculum=[
        dict(T=30, shape=(2, 2), min_digits=2, max_digits=3, dense_reward=True),
    ],
    base=10,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    n_train=10000,
    n_val=500,

    get_updater=get_updater,

    display_step=1000,
    eval_step=10,
    max_steps=100000,
    patience=np.inf,
    power_through=False,
    preserve_policy=True,

    slim=True,

    save_summaries=False,
    start_tensorboard=False,
    verbose=False,
    visualize=False,
    display=False,
    save_display=False,
    use_gpu=False,

    reward_window=0.1,
    threshold=0.05,

    name="QLearning",

    actor_config=Config(
        alg=QLearning,

        action_selection=epsilon_greedy,
        controller=DuelingLstmController(),
        double=True,

        test_time_explore=0.01,

        optimizer_spec="adam",

        init_steps=3000,

        target_update_rate=0.001,
        steps_per_target_update=None,
        patience=np.inf,
        batch_size=1,  # Number of sample experiences per update

        n_partitions=25,
        replay_max_size=1000,

        max_grad_norm=0.0,
    ),

    symbols=[
        # ('A', lambda x: sum(x)),
        ('M', lambda x: np.product(x)),
        # ('C', lambda x: len(x))
    ],
))


distributions = dict(
    n_controller_units=[32, 64, 128],
    exploration_schedule=[
        'Poly(1.0, 100000, end=0.01)',
        'Poly(1.0, 100000, end=0.1)',
        'Poly(10.0, 100000, end=0.01)',
        'Poly(10.0, 100000, end=0.1)',
    ],
    actor_config=dict(
        gamma=list(np.linspace(0.9, 1.0, 10)),
        opt_steps_per_update=[1, 5, 10],
        update_batch_size=[4, 8, 16, 32, 64],
        lr_schedule=['1e-5', '1e-4', '1e-3'],
        alpha=list(np.linspace(0.5, 1.0, 10)),
        beta_schedule=["Poly({}, 100000, end=1.0)".format(i) for i in [0.3, 0.5, 0.7, 0.9, 0.1]],
    ),
)

from ac_search import search
hosts = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i+1) for i in range(15)]
search(config, distributions, hosts=hosts)
