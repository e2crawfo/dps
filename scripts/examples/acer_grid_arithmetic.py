from dps.train import training_loop
from dps.config import DEFAULT_CONFIG
from dps.rl.algorithms import acer
from dps.envs import grid_arithmetic

config = DEFAULT_CONFIG.copy()

config.max_steps = 100000

acer_config = acer.config.copy(
    n_train=10000,
    n_val=100,
    optimizer_spec="adam",
    exploration_schedule=(
        "MixtureSchedule("
        "    [Poly(10, {0}, end=5.0), Poly(10, {0}, end=1.0), Poly(10, {0}, end=0.1)],"
        "    100, shared_clock=False)").format(config.max_steps),
    test_time_explore=-1,

    batch_size=16,

    policy_weight=1.0,

    min_experiences=500,
    replay_size=10000,
    replay_n_partitions=100,
    alpha=0.0,
    beta_schedule=0.0,

    lr_schedule=1e-4,
    n_controller_units=64,
    split=False,
    update_batch_size=16,
    updates_per_sample=10,
    opt_steps_per_update=1,
    epsilon=0.2,
    value_weight=10.0,
    entropy_weight='Poly(0.0625, {}, end=0.0)'.format(config.max_steps),
    lmbda=1.0,
    gamma=1.0,
    c=2,
    max_grad_norm=1.0,
)


env_config = grid_arithmetic.config.copy(
    symbols=[
        ('A', lambda x: sum(x)),
        # ('M', lambda x: np.product(x)),
        # ('C', lambda x: len(x)),
        # ('X', lambda x: max(x)),
        # ('N', lambda x: min(x))
    ],
    curriculum=[
        dict(T=40, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    mnist=False,
    force_2d=False,
)

config.update(acer_config)
config.update(env_config)

with config:
    training_loop()
