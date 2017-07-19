import argparse
from contextlib import ExitStack
import time

import tensorflow as tf

import clify

from dps import cfg
from dps.config import algorithms, tasks, test_configs
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
        controller = cfg.controller(action_selection.n_params, name="visualize")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('alg')
    parser.add_argument('task')
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")
    args, _ = parser.parse_known_args()

    task = [t for t in tasks if t.startswith(args.task)]
    assert len(task) == 1, "Ambiguity in task selection, possibilities are: {}.".format(task)
    task = task[0]

    _algorithms = list(algorithms) + ['visualize']
    alg = [a for a in _algorithms if a.startswith(args.alg)]
    assert len(alg) == 1, "Ambiguity in alg selection, possibilities are: {}.".format(alg)
    alg = alg[0]

    if args.pdb:
        with pdb_postmortem():
            _run(alg, task)
    else:
        _run(alg, task)


def _run(alg, task, _config=None, **kwargs):
    if alg == 'visualize':
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
        config.update(algorithms[alg])
        if _config is not None:
            config.update(_config)
        config.update(kwargs)

        with config:
            cl_args = clify.wrap_object(cfg).parse()
            cfg.update(cl_args)

            training_loop()
