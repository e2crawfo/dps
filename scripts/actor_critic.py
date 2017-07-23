import tensorflow as tf
from tensorflow.contrib.slim import fully_connected

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.experiments.room import Room
from dps.rl import TRPO, PPO, REINFORCE
from dps.rl.value import PolicyEvaluation, TrustRegionPolicyEvaluation, actor_critic
from dps.rl.policy import Deterministic, ProductDist, Normal, NormalWithFixedScale, Gamma
from dps.utils import CompositeCell, FeedforwardCell, MLP, Config


def build_env():
    return Room(cfg.T, cfg.reward_radius, cfg.max_step, cfg.restart_prob, cfg.dense_reward, cfg.l2l, cfg.n_val)


def get_updater(env):
    scale = 0.2
    action_selection = ProductDist(NormalWithFixedScale(scale), NormalWithFixedScale(scale))
    # action_selection = ProductDist(Normal(), Normal())

    # policy_controller = FeedforwardCell(
    #     lambda inp, output_size: fully_connected(inp, output_size, activation_fn=None),
    #     action_selection.n_params, name="actor_controller")

    # critic_controller = FeedforwardCell(
    #     lambda inp, output_size: fully_connected(inp, output_size, activation_fn=None),
    #     1, name="critic_controller")

    policy_controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
        MLP(),
        action_selection.n_params,
        name="actor_controller")

    critic_controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
        MLP(),
        1,
        name="critic_controller")

    if 1:
        critic_config = Config(
            critic_name="TRPE",
            critic_alg=TrustRegionPolicyEvaluation,
            delta_schedule='0.01',
            max_cg_steps=10,
            max_line_search_steps=10,
        )

    else:
        critic_config = Config(
            critic_name="PE",
            critic_alg=PolicyEvaluation,
            optimizer_spec='rmsprop',
            lr_schedule='1e-3',
        )

    if 0:
        actor_config = Config(
            actor_name="TRPO",
            actor_alg=TRPO,
            delta_schedule='0.01',
            max_cg_steps=10,
            max_line_search_steps=10,
            entropy_schedule='0.0',
            lmbda=1.0,
            gamma=1.0
        )
    elif 1:
        actor_config = Config(
            actor_name="PPO",
            actor_alg=PPO,
            epsilon=0.2,
            K=10,
            lr_schedule='1e-4',
            entropy_schedule='0.0',
            lmbda=1.0,
            gamma=1.0
        )
    else:
        actor_config = Config(
            actor_name="REINFORCE",
            actor_alg=REINFORCE,
            optimizer_spec='rmsprop',
            lr_schedule='1e-3',
            entropy_schedule='0.1',
            lmbda=1.0,
            gamma=1.0
        )

    return actor_critic(
        env, policy_controller, action_selection, critic_controller,
        actor_config, critic_config)


config = DEFAULT_CONFIG.copy(
    get_updater=get_updater,
    build_env=build_env,
    log_name="actor_critic",
    max_steps=100000,

    display_step=100,

    T=30,
    reward_radius=0.25,
    max_step=0.5,
    restart_prob=0.0,
    dense_reward=False,
    l2l=False,
    n_val=200,

    threshold=1e-4,

    verbose=False,
)

with config:
    cl_args = clify.wrap_object(cfg).parse()
    config.update(cl_args)

    val = training_loop()
