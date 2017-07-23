import numpy as np
import tensorflow as tf
import argparse

import clify

from dps import cfg
from dps.utils import Config, CompositeCell, MLP
from dps.parallel.submit_job import submit_job
from dps.parallel.hyper import build_search
from dps.rl import TRPO
from dps.rl.value import TrustRegionPolicyEvaluation, actor_critic


def get_updater(env):
    with cfg.actor_config:
        action_selection = cfg.action_selection()
        policy_controller = CompositeCell(
            tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
            MLP(),
            action_selection.n_params,
            name="actor_controller")

    with cfg.critic_config:
        critic_controller = CompositeCell(
            tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
            MLP(),
            1,
            name="critic_controller")

    return actor_critic(
        env, policy_controller, action_selection, critic_controller,
        cfg.actor_config, cfg.critic_config)


config = Config(
    curriculum=[
        dict(T=10, shape=(2, 2), n_digits=3, upper_bound=True),
        dict(T=15, shape=(3, 3), n_digits=3, upper_bound=True),
        dict(T=25, shape=(4, 4), n_digits=3, upper_bound=True),
        dict(T=30, shape=(5, 5), n_digits=3, upper_bound=True),
    ],
    base=10,
    gamma=0.99,
    upper_bound=True,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    n_train=10000,
    n_val=500,

    display_step=1000,
    eval_step=10,
    max_steps=100000,
    patience=5000,
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

    name="TrustRegionActorCritic",

    critic_config=Config(
        critic_name="TRPE",
        critic_alg=TrustRegionPolicyEvaluation,
        delta_schedule='0.01',
        max_cg_steps=10,
        max_line_search_steps=10,
    ),

    actor_config=Config(
        actor_name="TRPO",
        actor_alg=TRPO,
        max_cg_steps=10,
        max_line_search_steps=10,
    )
)


with config:
    cl_args = clify.wrap_object(cfg).parse()
    cfg.update(cl_args)

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
            entropy_schedule=[0.0] + list(0.5**np.arange(1, 4, step=1)) +
                             ['poly {} 100000 1e-6 1'.format(n) for n in 0.5**np.arange(1, 4, step=1)],
            delta_schedule=['1e-4', '1e-3', '1e-2'],
        ),
    )

    alg = 'trpo'
    task = 'alt_arithmetic'
    hosts = [":"]

    parser = argparse.ArgumentParser()
    parser.add_argument('size')
    parser.add_argument('walltime')
    args = parser.parse_args()
    walltime = args.walltime
    size = args.size

    if size == 'big':
        # Big
        n_param_settings = 50
        n_repeats = 5
        cleanup_time = "00:30:00"
        time_slack = 60
        ppn = 4
    elif size == 'medium':
        # Medium
        n_param_settings = 8
        n_repeats = 4
        cleanup_time = "00:02:15"
        time_slack = 30
        ppn = 4
    elif size == 'small':
        # Small
        n_param_settings = 2
        n_repeats = 2
        cleanup_time = "00:00:45"
        time_slack = 10
        ppn = 2
    else:
        raise Exception()

    job, archive_path = build_search(
        '/tmp/dps/search', 'ac_trpo_search', n_param_settings, n_repeats,
        alg, task, True, distributions, config, use_time=1)

    submit_job(
        "AC_TRPO_SEARCH", archive_path, 'map', '/tmp/dps/search/execution/',
        parallel_exe='$HOME/.local/bin/parallel', dry_run=False,
        env_vars=dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1'), ppn=ppn, hosts=hosts,
        walltime=walltime, cleanup_time=cleanup_time, time_slack=time_slack, redirect=True)
