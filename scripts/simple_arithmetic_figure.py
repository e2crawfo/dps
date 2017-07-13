from dps.utils import DpsConfig, MLP, CompositeCell
from dps.experiments.simple_arithmetic import render_rollouts_static
from dps.vision import LeNet
from dps.run import _run
from dps.rl.policy import Softmax
import tensorflow as tf


n_examples = 8

config = DpsConfig()
config.update({
    'base': 10,
    'build_classifier': None,
    'classifier_str': 'LeNet_256',
    'T': 12,
    'n_digits': 3,
    'op_loc': (0, 0),
    'start_loc': (0, 0),
    'shape': (3, 3),
    'upper_bound': True,
    'display': True,
    'save_display': True,
    'mnist': 1,
    'mpl_backend': 'TkAgg',
    'n_controller_units': 128,
    'n_train': n_examples,
    'n_val': n_examples,
    'batch_size': n_examples,
    'reward_window': 0.5,
    'start_tensorboard': False,
    'test_time_explore': 0.1,
    'use_gpu': 0,
    'verbose': 1,
    'visualize': True,
    'policy_scope': 'SimpleArithmetic_policy',
    'render_rollouts': render_rollouts_static,
    'controller_func': lambda n_actions: CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=128), MLP(), n_actions)
    })


def build_classifier(inp, output_size, is_training=False):
    logits = LeNet(256, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)
    return tf.nn.softmax(logits)


config.build_classifier = build_classifier
config.action_selection = Softmax()

load_from = "/home/eric/Dropbox/projects/dps/good/solves_5x5_3digits/best_stage=2"

_run("visualize", "simple_arithmetic", config, load_from)
