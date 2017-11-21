import numpy as np
import clify
import tensorflow as tf
import argparse

from config import rl_config as config

from dps import cfg
from dps.rl.policy import Policy, EpsilonSoftmax, Normal, ProductDist, Deterministic
from dps.utils.tf import FullyConvolutional, LeNet, CompositeCell, MLP

parser = argparse.ArgumentParser()
parser.add_argument("ablation", choices="no_modules no_classifiers no_ops".split())
args = parser.parse_args()

if args.ablation == 'no_modules':
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
        return Policy(
            ProductDist(
                EpsilonSoftmax(4, one_hot=False),
                Deterministic(env.n_classes),
            ),
            env.obs_shape, **kwargs
        )

    config.update(
        ablation='no_modules',
        build_policy=build_policy,
        build_controller=BuildNoModulesController(),
    )

elif args.ablation == 'no_classifiers':
    # B
    pass

elif args.ablation == 'no_ops':
    # C
    pass
else:
    raise Exception("NotImplemented")

config.update(ablations=args.ablation)


grid = dict(n_train=2**np.arange(6, 18, 2))


from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
