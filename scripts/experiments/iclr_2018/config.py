from dps import cfg
from dps.utils import Config
from dps.utils.tf import LeNet
from dps.envs import grid_arithmetic
from dps.rl.algorithms import a2c
from dps.rl.policy import BuildEpsilonSoftmaxPolicy, BuildLstmController
from dps.config import RL_EXPERIMENT_CONFIG, SL_EXPERIMENT_CONFIG
from dps.updater import DifferentiableUpdater
from dps.environment import RegressionEnv


env_config = grid_arithmetic.config.copy(
    reductions="A:sum,M:prod,X:max,N:min",
    arithmetic_actions='+,*,max,min,+1',
    ablation='easy',
    render_rollouts=None,

    curriculum=[
        dict(),
    ],
    T=30,
    min_digits=2,
    max_digits=3,
    shape=(2, 2),
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.01,
    largest_digit=100,

    salience_action=True,
    visible_glimpse=False,
    initial_salience=False,
    salience_input_shape=(3*14, 3*14),
    salience_output_shape=(14, 14),
    image_shape=(14, 14),

    final_reward=True,

    n_train=10000,
    n_val=100,
)

alg_config = Config(
    get_updater=a2c.A2C,
    build_policy=BuildEpsilonSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",

    policy_weight=1.0,
    entropy_weight=0.01,

    value_weight=1.0,
    value_reg_weight=0.0,
    value_epsilon=0,
    value_n_samples=0,
    value_direct=False,

    lr_schedule=1e-4,
    n_controller_units=128,
    batch_size=16,
    gamma=0.98,
    opt_steps_per_update=1,
    epsilon=0.2,
    split=False,

    exploration_schedule="Poly(1.0, 0.1, 8192)",
    actor_exploration_schedule=None,
    val_exploration_schedule="0.0",

    q_lmbda=1.0,
    v_lmbda=1.0,
    policy_importance_c=0,
    q_importance_c=None,
    v_importance_c=None,
    max_grad_norm=None,

    updates_per_sample=1,
)

rl_config = RL_EXPERIMENT_CONFIG.copy(
    name="GridArithmeticRL",

    memory_limit_mb=12*1024,
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,
)

rl_config.update(alg_config)
rl_config.update(env_config)


def sl_build_env():
    train = grid_arithmetic.GridArithmeticDataset(n_examples=cfg.n_train)
    val = grid_arithmetic.GridArithmeticDataset(n_examples=cfg.n_val)
    test = grid_arithmetic.GridArithmeticDataset(n_examples=cfg.n_val)
    return RegressionEnv(train, val, test)


def get_updater(env):
    build_model = LeNet(n_units=int(cfg.n_controller_units))
    return DifferentiableUpdater(env, build_model)


cnn_config = SL_EXPERIMENT_CONFIG.copy(
    name="GridArithmeticCNN",

    memory_limit_mb=12*1024,
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,

    get_updater=get_updater,
    optimizer_spec="adam",
    lr_schedule=1e-4,
    power_through=True,
    noise_schedule=0.0,
    max_grad_norm=None,
    l2_weight=0.0,

    batch_size=64,
    n_controller_units=128,
    log_name="cnn_grid_arithmetic",
    patience=5000,
    reward_window=0.499,
    one_hot_output=True,
    loss_type="xent",
    stopping_function=lambda val_record: -val_record['reward'],
    preserve_policy=True,
)

cnn_config.update(env_config)
