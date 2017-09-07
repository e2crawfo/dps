import numpy as np

from dps.utils import Config
from dps.rl import PPO
from dps.rl.value import TrustRegionPolicyEvaluation, ProximalPolicyEvaluation, PolicyEvaluation
from dps.config import get_updater, tasks, LstmController, softmax


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
    action_selection=softmax,
    controller=LstmController(),
    n_controller_units=128,
    classification_bonus=0.0,

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
    exploration_schedule='Poly(10.0, 100000, end=0.1)',
    batch_size=16,
    test_time_explore=1.0,

    name="PPOActorCritic",

    actor_config=Config(
        name="PPO",
        alg=PPO,
        opt_steps_per_update=10,
        optimizer_spec="adam",
        gamma=0.9,
        lmbda=0.9,
        entropy_schedule="0.0625",
    ),

    critic_config=Config(
        optimizer_spec='adam',
        max_line_search_steps=10,
        S=1,
        opt_steps_per_update=10,
        max_cg_steps=10,
    ),

    symbols=[
        ('A', lambda x: sum(x)),
        ('M', lambda x: np.product(x)),
        ('C', lambda x: len(x))
    ],
))

distributions = dict(

    actor_config=dict(
        epsilon=[0.1, 0.2, 0.3],
        lr_schedule=[1e-5, 1e-4, 1e-3],
    ),

    critic_config=dict(
        alg=["TRPE", "PPE", "PE"],

        # For TRPE
        delta_schedule=[1e-3, 1e-2, 1e-1],

        # For PPE and PE
        lr_schedule=[1e-5, 1e-4, 1e-3],

        # For PPE
        epsilon=[0.1, 0.2, 0.3],
    )
)

from ac_search import search
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
# hosts = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in [16, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30, 32]]
search(config, distributions, host_pool=host_pool)
