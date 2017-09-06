import numpy as np

from dps.utils import Config
from dps.envs import grid_arithmetic
from dps.rl.algorithms import acer
from dps.rl.policy import BuildSoftmaxPolicy, BuildLstmController


config = Config(
    name="AcerSearch",

    n_train=10000,
    n_val=500,
    max_steps=10000,
    display_step=1000,
    eval_step=10,
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
)


alg_config = Config(
    get_updater=acer.ACER,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",

    exploration_schedule="Poly(10.0, 10000, 0.1)",
    test_time_explore=0.1,

    batch_size=1,
    opt_steps_per_update=1,

    policy_weight=1.0,

    min_experiences=1000,
    replay_size=10000,
    replay_n_partitions=100,
    alpha=0.0,
    beta_schedule=0.0,
)


env_config = Config(
    build_env=grid_arithmetic.build_env,
    symbols=[
        ('A', lambda x: sum(x)),
        # ('M', lambda x: np.product(x)),
        # ('C', lambda x: len(x))
    ],

    curriculum=[
        dict(T=40, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    force_2d=False,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.04,
    classification_bonus=0.0,

    dense_reward=True,
    reward_window=0.4,

    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.
    log_name='grid_arithmetic',
    render_rollouts=None
)


config.update(alg_config)
config.update(env_config)


distributions = dict(
    lr_schedule=['Poly({}, 10000, end=1e-6)'.format(i) for i in [1e-3, 1e-4, 1e-5]],
    n_controller_units=[32, 64, 128],
    split=[True, False],
    update_batch_size=[1, 8, 32],
    updates_per_sample=[1, 8, 32],
    epsilon=[0.1, 0.2, 0.3],
    value_weight=2**np.arange(4),
    entropy_weight=[0.0] + ['Poly({}, 10000, end=0.0)'.format(i) for i in 0.5**np.arange(1, 5)],
    lmbda=list(np.linspace(0.8, 1.0, 10)),
    gamma=list(np.linspace(0.9, 1.0, 10)),
    c=[1, 2, 4, 8],
)


from dps.parallel.hyper import build_and_submit_search
hosts = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 17)]
# hosts = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in [16, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30, 32]]
build_and_submit_search(config, distributions, hosts=hosts)
