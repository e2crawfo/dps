import argparse
from contextlib import ExitStack
import time

import tensorflow as tf

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop, gen_seed, uninitialized_variables_initializer
from dps.utils import pdb_postmortem
from dps.rl.policy import Policy
from dps.rl import algorithms as algorithms_module
from dps import envs as envs_module


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


def parse_env_alg(env, alg):
    envs = [e for e in dir(envs_module) if e.startswith(env)]
    assert len(envs) == 1, "Ambiguity in env selection, possibilities are: {}.".format(env)
    env_config = getattr(envs_module, envs[0]).config

    algs = [a for a in dir(algorithms_module) if a.startswith(alg)]
    assert len(algs) == 1, "Ambiguity in alg selection, possibilities are: {}.".format(alg)
    alg_config = getattr(algorithms_module, algs[0]).config

    return env_config, alg_config


def run():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('env')
    parser.add_argument('alg')
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")
    args, _ = parser.parse_known_args()

    env, alg = parse_env_alg(args.env, args.alg)

    if args.pdb:
        with pdb_postmortem():
            _run(env, alg)
    else:
        _run(env, alg)


def _run(env_config, alg_config, _config=None, visualize=False, **kwargs):
    if visualize:
        raise Exception("NotImplemented")
        config = None
        if _config is not None:
            config.update(_config)
        config.update(display=True, save_display=True)
        config.update(kwargs)

        with config:
            cl_args = clify.wrap_object(cfg).parse()
            config.update(cl_args)

            build_and_visualize()
    else:
        config = DEFAULT_CONFIG.copy()
        config.update(alg_config)
        config.update(env_config)

        if _config is not None:
            config.update(_config)
        config.update(kwargs)

        with config:
            cl_args = clify.wrap_object(cfg).parse()
            cfg.update(cl_args)

            training_loop()
