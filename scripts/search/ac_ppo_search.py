import numpy as np

from dps.utils import Config
from dps.rl import PPO
from dps.rl.value import TrustRegionPolicyEvaluation
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

    name="PPOActorCritic",

    # critic_config=Config(
    #     name="PPE",
    #     alg=ProximalPolicyEvaluation,
    #     optimizer_spec='adam',
    #     opt_steps_per_update=10,
    # ),

    critic_config=Config(
        name="TRPE",
        alg=TrustRegionPolicyEvaluation,
        max_cg_steps=10,
        max_line_search_steps=10,
    ),

    actor_config=Config(
        name="PPO",
        alg=PPO,
        opt_steps_per_update=10,
        optimizer_spec="adam"
    ),
    symbols=[
        ('A', lambda x: sum(x)),
        ('M', lambda x: np.product(x)),
        # ('C', lambda x: len(x))
    ],
))


distributions = dict(
    n_controller_units=[32, 64, 128],
    classification_bonus=list(np.linspace(0.0, 0.1, 10)),
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
    # critic_config=dict(
    #     epsilon=[0.1, 0.2, 0.3, 0.4],
    #     lr_schedule=['1e-5', '1e-4', '1e-3', '1e-2'],
    #     S=[1, 4, 8]
    # ),
    actor_config=dict(
        lmbda=list(np.linspace(0.8, 1.0, 10)),
        gamma=list(np.linspace(0.9, 1.0, 10)),
        entropy_schedule=[0.0] + list(0.5**np.arange(1, 5)) +
                         ['poly {} 100000 1e-6 1'.format(n) for n in 0.5**np.arange(1, 5)],
        lr_schedule=['1e-5', '1e-4', '1e-3'],
        epsilon=[0.1, 0.2, 0.3]
    ),
)

from ac_search import search
hosts = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 17)]
# hosts = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in [16, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30, 32]]
search(config, distributions, hosts=hosts)
