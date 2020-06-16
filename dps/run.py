import argparse
import pkgutil

from dps import cfg
from dps.train import training_loop
from dps.utils import ipdb_postmortem, get_default_config
from dps.rl import algorithms as alg_pkg
import dps.env.basic as env_pkg_basic


def get_module_specs(*packages):
    specs = {}
    for p in packages:
        update = {
            name: (loader, name, is_pkg)
            for loader, name, is_pkg in pkgutil.iter_modules(p.__path__)
        }

        intersection = specs.keys() & update.keys()
        assert not intersection, \
            "Module name overlaps: {}".format(list(intersection))
        specs.update(update)
    return specs


def get_module_from_spec(spec):
    return spec[0].find_module(spec[1]).load_module(spec[1])


def parse_env_alg(env, alg=None):
    env_module_specs = get_module_specs(env_pkg_basic)  # , env_pkg_nips_2018)

    if "." in env:
        env_file, env_suffix = env.split('.')
    else:
        env_file, env_suffix = env, ""

    env_spec = env_module_specs.get(env_file, None)
    if env_spec is None:
        candidates = [e for e in env_module_specs.keys() if e.startswith(env_file)]
        assert len(candidates) == 1, \
            "Ambiguity in env selection, possibilities are: {}.".format(candidates)
        env_spec = env_module_specs[candidates[0]]

    env_module = get_module_from_spec(env_spec)
    if env_suffix:
        env_config = getattr(env_module, "{}_config".format(env_suffix))
    else:
        env_config = env_module.config

    if alg is None:
        alg_config = {}
    else:
        config_name = "{}_config".format(alg)
        alg_config = getattr(env_module, config_name, None)

        if alg_config is None:
            alg_module_specs = get_module_specs(alg_pkg)

            if "." in alg:
                alg_file, alg_suffix = alg.split('.')
            else:
                alg_file, alg_suffix = alg, ""

            alg_spec = alg_module_specs.get(alg_file, None)

            if alg_spec is None:
                candidates = [a for a in alg_module_specs.keys() if a.startswith(alg_file)]
                assert len(candidates) == 1, (
                    "Ambiguity in alg selection, possibilities "
                    "are: {}.".format(candidates))

                alg_spec = alg_module_specs[candidates[0]]

            alg_module = get_module_from_spec(alg_spec)
            if alg_suffix:
                alg_config = getattr(alg_module, "{}_config".format(alg_suffix))
            else:
                alg_config = alg_module.config

    return env_config, alg_config


def run():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('env', default=None, help="Name (or unique name-prefix) of environment to run.")
    parser.add_argument('alg', nargs='?', default=None,
                        help="Name (or unique name-prefix) of algorithm to run. Optional. "
                             "If not provided, algorithm spec is assumed to be included "
                             "in the environment spec.")
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")
    args, _ = parser.parse_known_args()

    env = args.env
    alg = args.alg

    # This can happen by accident when there is another argument passed of the form --x="y z"
    # that is not parsed by the current parser (since we are using `parse_known_args`).
    if isinstance(env, str) and env.startswith("--"):
        env = None

    if isinstance(alg, str) and alg.startswith("--"):
        alg = None

    if args.pdb:
        with ipdb_postmortem():
            _run(env, alg)
    else:
        _run(env, alg)


def _run(env_str, alg_str, _config=None, **kwargs):
    env_config, alg_config = parse_env_alg(env_str, alg_str)

    config = get_default_config()
    config.update(alg_config)
    config.update(env_config)

    if _config is not None:
        config.update(_config)
    config.update(kwargs)

    with config:
        cfg.update_from_command_line()
        return training_loop()


def _raw_run(config):
    with config:
        return training_loop()
