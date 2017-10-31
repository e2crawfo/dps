import numpy as np
import clify
import tensorflow as tf

from dps import cfg
from dps.utils import Config
from dps.envs import grid_arithmetic
from dps.rl.algorithms import a2c
from dps.rl.policy import Policy, EpsilonSoftmax, Normal, ProductDist, BuildLstmController
from dps.rl import rl_render_hook
from dps.config import DEFAULT_CONFIG
from dps.utils.tf import FullyConvolutional, LeNet, CompositeCell, MLP


config = DEFAULT_CONFIG.copy(
    name="GridArithmeticExperiments",

    n_train=10000,
    n_val=100,
    max_steps=1000000,
    display_step=100,
    eval_step=100,
    patience=np.inf,
    power_through=True,
    preserve_policy=True,

    slim=False,
    save_summaries=True,
    start_tensorboard=True,
    verbose=False,
    show_plots=True,
    save_plots=True,

    threshold=0.01,
    render_hook=rl_render_hook,
    memory_limit_mb=12*1024,
)


alg_config = Config(
    get_updater=a2c.A2C,
    optimizer_spec="adam",

    policy_weight=1.0,
    value_reg_weight=0.0,
    value_weight=1.0,
    entropy_weight=0.01,

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

env_config = grid_arithmetic.config.copy(
    reductions="sum",
    arithmetic_actions='+,*,max,min,+1',
    ablation='easy',
    render_rollouts=None,

    T=30,
    min_digits=2,
    max_digits=3,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.01,

    salience_shape=(2, 2),
    salience_action=True,
    visible_glimpse=False,
    initial_salience=False,
    salience_input_width=3*14,
    salience_output_width=14,
    downsample_factor=2,

    final_reward=True,
)

config.update(alg_config)
config.update(env_config)

ablation = 'no_modules'

if ablation == 'no_modules':
    # A
    def no_modules_inp(obs):
        glimpse_start = 4 + 14**2
        glimpse_end = glimpse_start + 14 ** 2
        glimpse = obs[..., glimpse_start:glimpse_end]
        glimpse_processor = FullyConvolutional(
            [
                dict(num_outputs=16, kernel_size=3, activation_fn=tf.nn.relu, padding='same'),
                dict(num_outputs=16, kernel_size=3, activation_fn=tf.nn.relu, padding='same'),
            ],
            pool=True,
            flatten_output=True
        )
        glimpse_features = glimpse_processor(glimpse, 1, False)
        return tf.concat(
            [obs[..., :glimpse_start], glimpse_features, obs[..., glimpse_end:]],
            axis=-1
        )

    class BuildNoModulesController(object):
        def __call__(self, params_dim, name=None):
            return CompositeCell(
                tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
                MLP(), params_dim, inp=no_modules_inp, name=name)

    def build_policy(env, **kwargs):
        # return Policy(
        #     ProductDist(EpsilonSoftmax(5, one_hot=True), Normal()),
        #     env.obs_shape, **kwargs)

        # But maybe this should actually just be differentiable. Could train it using cross entropy.
        # I'm actually quite interested to see if we could make this work. Would be a more direct extension of RMVA.
        return Policy(
            ProductDist(EpsilonSoftmax(5, one_hot=True), EpsilonSoftmax(one_hot=False)),
            env.obs_shape, **kwargs)

    config.update(
        ablation='no_modules',
        build_policy=build_policy,
        build_controller=BuildNoModulesController(),
        n_lenet_units=128,
        n_lenet_outputs=128,
    )
elif ablation == 'no_classifiers':
    # B
    config.update(
        ablation='no_classifiers',
    )
else:
    # C
    config.update(
        ablation='no_ops',
    )


config.update(
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,
    load_path="",
)

grid = dict(n_train=2**np.arange(6, 18))


from dps.parallel.hyper import build_and_submit
host_pool = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=host_pool)
