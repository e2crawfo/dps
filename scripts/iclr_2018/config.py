from dps import cfg
from dps.utils import Config
from dps.utils.tf import LeNet, MLP
from dps.env import grid_arithmetic
from dps.rl.algorithms import a2c
from dps.rl.policy import BuildEpsilonSoftmaxPolicy, BuildLstmController
from dps.config import RL_EXPERIMENT_CONFIG, SL_EXPERIMENT_CONFIG
from dps.vision.train import SALIENCE_CONFIG, EMNIST_CONFIG


env_config = Config(
    log_name='grid_arithmetic',
    render_rollouts=grid_arithmetic.config.render_rollouts,
    build_env=grid_arithmetic.config.build_env,
    build_policy=grid_arithmetic.config.build_policy,

    reductions="A:sum,M:prod,X:max,N:min",
    arithmetic_actions="+,*,max,min,+1",

    curriculum=[dict()],
    base=10,
    threshold=0.04,
    T=30,
    min_digits=2,
    max_digits=3,
    final_reward=True,
    parity='both',
    reward_window=0.4999,

    op_loc=(0, 0),  # With respect to draw_shape_grid
    start_loc=(0, 0),  # With respect to env_shape_grid
    image_shape_grid=(2, 2),
    draw_offset=(0, 0),
    draw_shape_grid=None,
    patch_shape=(14, 14),

    salience_action=True,
    visible_glimpse=False,
    salience_input_shape=(3*14, 3*14),
    salience_output_shape=(14, 14),
    initial_salience=False,
    salience_model=True,

    n_train=10000,
    n_val=100,
    max_steps=300001,
    stopping_criteria_name="01_loss,min",
    use_gpu=False,

    ablation='easy',

    build_digit_classifier=lambda: LeNet(128, scope="digit_classifier"),
    build_op_classifier=lambda: LeNet(128, scope="op_classifier"),
    build_omniglot_classifier=lambda: LeNet(128, scope="omniglot_classifier"),

    emnist_config=EMNIST_CONFIG.copy(),
    salience_config=SALIENCE_CONFIG.copy(
        min_digits=0,
        max_digits=4,
        std=0.05,
        n_units=100
    ),

    largest_digit=1000,

    n_glimpse_features=128,
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

    use_differentiable_loss=False,
)

rl_config = RL_EXPERIMENT_CONFIG.copy()

rl_config.update(alg_config)
rl_config.update(env_config)

rl_config.update(
    name="GridArithmeticRL",

    cpu_ram_limit_mb=12*1024,
    use_gpu=False,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,
)


cnn_config = SL_EXPERIMENT_CONFIG.copy()

cnn_config.update(env_config)
cnn_config.update(
    name="GridArithmeticCNN",

    cpu_ram_limit_mb=12*1024,
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,

    stopping_criteria="01_loss,min",
    get_updater=grid_arithmetic.sl_get_updater,
    optimizer_spec="adam",
    lr_schedule=1e-4,
    power_through=True,
    noise_schedule=0.0,
    max_grad_norm=None,
    l2_weight=0.0,

    batch_size=64,
    log_name="cnn_grid_arithmetic",
    patience=5000,
    reward_window=0.499,
    load_path=-1,

    build_env=grid_arithmetic.sl_build_env,
    build_model=grid_arithmetic.feedforward_build_model,

    mode="standard",
    n_controller_units=128,
    build_convolutional_model=lambda: LeNet(cfg.n_controller_units),

    largest_digit=99,

    # For when mode == "pretrained"
    fixed=True,
    pretrain=True,
    build_feedforward_model=lambda: MLP(
        [cfg.n_controller_units, cfg.n_controller_units, cfg.n_controller_units]
    ),
    n_raw_features=128,
)


rnn_config = SL_EXPERIMENT_CONFIG.copy()

rnn_config.update(env_config)
rnn_config.update(
    name="GridArithmeticRNN",

    cpu_ram_limit_mb=12*1024,
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,

    stopping_criteria="01_loss,min",
    get_updater=grid_arithmetic.sl_get_updater,
    optimizer_spec="adam",
    lr_schedule=1e-4,
    power_through=True,
    noise_schedule=0.0,
    max_grad_norm=None,
    l2_weight=0.0,

    batch_size=64,
    log_name="rnn_grid_arithmetic",
    patience=5000,
    reward_window=0.499,
    load_path=-1,

    build_env=grid_arithmetic.sl_build_env,
    build_model=grid_arithmetic.recurrent_build_model,

    largest_digit=99,

    n_controller_units=128,
    build_recurrent_model=BuildLstmController(),

    pretrain=True,

    n_raw_features=128,
    build_convolutional_model=lambda: LeNet(cfg.n_raw_features),

    # For when pretrain == True
    fixed=True,

    # For when pretrain == False
    n_glimpse_features=64,
    build_glimpse_processor=lambda: LeNet(cfg.n_glimpse_features),
)
