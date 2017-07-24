import numpy as np
import tensorflow as tf

from dps import cfg
from dps.utils import Config, CompositeCell, MLP
from dps.rl.policy import Softmax
from dps.rl import REINFORCE
from dps.rl.value import TrustRegionPolicyEvaluation
from dps.config import get_updater, tasks


config = tasks['alt_arithmetic']


config.update(Config(
    curriculum=[
        dict(T=30, shape=(2, 2), n_digits=3, upper_bound=True),
    ],
    base=10,
    gamma=0.99,
    upper_bound=True,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    n_train=10000,
    n_val=500,

    get_updater=get_updater,
    action_selection=lambda env: Softmax(env.n_actions),
    controller=lambda n_params, name: CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
        MLP(),
        n_params,
        name=name
    ),

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

    noise_schedule=None,

    name="ReinforceActorCritic",

    critic_config=Config(
        name="TRPE",
        alg=TrustRegionPolicyEvaluation,
        delta_schedule='0.01',
        max_cg_steps=10,
        max_line_search_steps=10,
    ),

    actor_config=Config(
        name="REINFORCE",
        alg=REINFORCE,
    )
))

distributions = dict(
    n_controller_units=[32, 64, 128],
    batch_size=[16, 32, 64, 128],
    exploration_schedule=[
        'poly 1.0 100000 0.01',
        'poly 1.0 100000 0.1',
        'poly 10.0 100000 0.01',
        'poly 10.0 100000 0.1',
    ],
    test_time_explore=[1.0, 0.1, -1],
    critic_config=dict(
        delta_schedule=['1e-3', '1e-2'],
    ),
    actor_config=dict(
        lmbda=list(np.linspace(0.8, 1.0, 10)),
        gamma=list(np.linspace(0.9, 1.0, 10)),
        entropy_schedule=[0.0] + list(0.5**np.arange(2, 5, step=1)) +
                         ['poly {} 100000 1e-6 1'.format(n) for n in 0.5**np.arange(2, 5, step=1)],
        optimizer_spec=["rmsprop 0.95 0.95 1e-8", "rmsprop 0.9 0.0 1e-10", "adam"],
        lr_schedule=[1e-3, 1e-4, 1e-5]
    )
)

from ac_search import search
search(config, distributions)
