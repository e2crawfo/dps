import argparse
from contextlib import ExitStack
import time

import tensorflow as tf

import clify

from dps import cfg
from dps.config import parse_task_actor_critic, tasks, actor_configs, critic_configs, test_configs
from dps.train import training_loop, gen_seed, uninitialized_variables_initializer
from dps.utils import pdb_postmortem
from dps.rl.policy import Policy


def build_and_visualize():
    with ExitStack() as stack:
        graph = tf.Graph()

        if not cfg.use_gpu:
            stack.enter_context(graph.device("/cpu:0"))

        sess = tf.Session(graph=graph)

        stack.enter_context(graph.as_default())
        stack.enter_context(sess)
        stack.enter_context(sess.as_default())

        tf_seed = gen_seed()
        tf.set_random_seed(tf_seed)

        env = cfg.build_env()

        exploration = tf.constant(cfg.test_time_explore or 0.0)

        action_selection = cfg.action_selection(env)
        controller = cfg.controller(action_selection.params_dim, name="visualize")
        policy = Policy(controller, action_selection, env.obs_shape)
        policy.set_exploration(exploration)

        policy_scope = getattr(cfg, 'policy_scope', None)
        if policy_scope:
            with tf.variable_scope(policy_scope) as scope:
                policy.set_scope(scope)

        try:
            sess.run(uninitialized_variables_initializer())
            sess.run(tf.assert_variables_initialized())
        except TypeError:
            pass

        start_time = time.time()

        render_rollouts = getattr(cfg, 'render_rollouts', None)
        env.visualize(
            policy=policy, n_rollouts=cfg.batch_size,
            T=cfg.T, mode='train', render_rollouts=render_rollouts)

        duration = time.time() - start_time

        print("Visualization took {} seconds.".format(duration))


def run():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('task')
    parser.add_argument('actor')
    parser.add_argument('--critic', type=str, default="baseline")
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")
    args, _ = parser.parse_known_args()

    task, actor, critic = parse_task_actor_critic(args.task, args.actor, args.critic)

    if args.pdb:
        with pdb_postmortem():
            _run(task, actor, critic)
    else:
        _run(task, actor, critic)


def _run(task, actor, critic, _config=None, **kwargs):
    if actor == 'visualize':
        config = test_configs[task]
        if _config is not None:
            config.update(_config)
        config.update(display=True, save_display=True)
        config.update(kwargs)

        with config:
            cl_args = clify.wrap_object(cfg).parse()
            config.update(cl_args)

            build_and_visualize()
    else:
        config = tasks[task]
        config.actor_config = actor_configs[actor]
        config.critic_config = critic_configs[critic]

        if _config is not None:
            config.update(_config)
        config.update(kwargs)

        with config:
            cl_args = clify.wrap_object(cfg).parse()
            cfg.update(cl_args)

            training_loop()
