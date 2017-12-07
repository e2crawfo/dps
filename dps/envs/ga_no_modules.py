import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.supervised import ClassificationEnv
from dps.environment import CompositeEnv
from dps.utils.tf import LeNet, MLP, CompositeCell
from dps.utils import Param, Config
from dps.rl.policy import EpsilonSoftmax, ProductDist, Policy, Deterministic

from dps.datasets import GridArithmeticDataset
from dps.envs.grid_arithmetic import GridArithmetic, render_rollouts
from dps.envs.grid_arithmetic import config as ga_config


def build_env():
    internal = GridArithmeticNoModules()

    train = GridArithmeticDataset(n_examples=cfg.n_train, one_hot=False)
    val = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=False)
    test = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=False)
    external = ClassificationEnv(train, val, test, one_hot=False)

    env = CompositeEnv(external, internal)
    env.obs_is_image = True
    return env


def build_policy(env, **kwargs):
    action_selection = ProductDist(
        EpsilonSoftmax(env.n_discrete_actions, one_hot=True),
        Deterministic(cfg.largest_digit+2))
    return Policy(action_selection, env.obs_shape, **kwargs)


def no_modules_inp(obs):
    glimpse_start = 3 + 14**2
    glimpse_end = glimpse_start + 14 ** 2
    glimpse = obs[..., glimpse_start:glimpse_end]
    glimpse_processor = LeNet(cfg.n_glimpse_units, scope="glimpse_classifier")
    glimpse_features = glimpse_processor(glimpse, cfg.n_glimpse_features, False)
    return tf.concat(
        [obs[..., :glimpse_start], glimpse_features, obs[..., glimpse_end:]],
        axis=-1
    )


def build_controller(params_dim, name=None):
    return CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units),
        MLP(), params_dim, inp=no_modules_inp, name=name)


config_delta = Config(
    log_name='grid_arithmetic_no_modules',
    render_rollouts=render_rollouts,
    build_env=build_env,
    build_policy=build_policy,
    build_controller=build_controller,
    largest_digit=99,
    n_glimpse_features=128,
    n_glimpse_units=128,
    use_differentiable_loss=True,
)


config = ga_config.copy()
config.update(config_delta)


class GridArithmeticNoModules(GridArithmetic):
    has_differentiable_loss = True
    _action_names = ['>', '<', 'v', '^', 'update_salience', 'output']

    largest_digit = Param()

    def __init__(self, **kwargs):
        super(GridArithmeticNoModules, self).__init__()

        self.action_names = self._action_names
        self.n_classes = self.largest_digit + 2
        self.action_sizes = [1, 1, 1, 1, 1, self.n_classes]
        self.actions_dim = sum(self.action_sizes)

    @property
    def n_discrete_actions(self):
        return 5

    def build_reward(self, registers, actions):
        loss = tf.cond(
            self.is_testing,
            lambda: tf.zeros(tf.shape(registers)[:-1])[..., None],
            lambda: tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.to_int32(self.targets_one_hot),
                logits=actions[-1])[..., None]
        )
        rewards = -loss
        rewards /= tf.to_float(self.T)
        return rewards

    def build_trajectory_loss(self, actions, visible, hidden):
        """ Compute loss for an entire trajectory. """
        logits = actions[..., self.n_discrete_actions:]
        targets = self.rb.get_from_hidden("y", hidden)
        targets = tf.one_hot(tf.cast(tf.squeeze(targets, axis=-1), tf.int32), self.n_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)[..., None]
        T_int = tf.shape(visible)[0]
        T = tf.to_float(T_int)
        multiplier = tf.one_hot(T_int-1, T_int)
        multiplier = multiplier + 1 / T * tf.ones_like(multiplier)
        multiplier = multiplier[:, None, None]
        loss *= multiplier
        return loss

    def build_init(self, r):
        self.maybe_build_placeholders()
        self.targets_one_hot = tf.one_hot(tf.cast(tf.squeeze(self.target_ph, axis=-1), tf.int32), self.n_classes)

        _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input, _y = self.rb.as_tuple(r)

        batch_size = tf.shape(self.input_ph)[0]

        # init fovea
        if self.start_loc is not None:
            fovea_y = tf.fill((batch_size, 1), self.start_loc[0])
            fovea_x = tf.fill((batch_size, 1), self.start_loc[1])
        else:
            fovea_y = tf.random_uniform(
                tf.shape(fovea_y), 0, self.env_shape[0], dtype=tf.int32)
            fovea_x = tf.random_uniform(
                tf.shape(fovea_x), 0, self.env_shape[1], dtype=tf.int32)

        fovea_y = tf.cast(fovea_y, tf.float32)
        fovea_x = tf.cast(fovea_x, tf.float32)

        glimpse = self._build_update_glimpse(fovea_y, fovea_x)

        salience = _salience
        salience_input = _salience_input
        if self.initial_salience:
            salience, salience_input = self._build_update_salience(
                1.0, _salience, _salience_input, _fovea_y, _fovea_x)

        return self.rb.wrap(fovea_x, fovea_y, _prev_action, salience, glimpse, salience_input, self.target_ph)

    def _init_networks(self):
        self.maybe_build_salience_detector()

    def _init_rb(self):
        values = (
            [0., 0., -1.] +
            [np.zeros(self.salience_output_size, dtype='f')] +
            [np.zeros(self.sub_image_size, dtype='f')] +
            [np.zeros(self.salience_input_size, dtype='f')] +
            [0.]
        )

        self.rb = RegisterBank(
            'GridArithmeticNoModulesRB',
            'fovea_x fovea_y prev_action salience glimpse', 'salience_input y', values=values,
            no_display='glimpse salience salience_input y',
        )

    def build_step(self, t, r, a):
        _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input, _y = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        right, left, down, up, update_salience, output = actions

        fovea_y, fovea_x = self._build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)
        glimpse = self._build_update_glimpse(fovea_y, fovea_x)

        salience = _salience
        salience_input = _salience_input
        if self.salience_action:
            salience, salience_input = self._build_update_salience(
                update_salience, _salience, _salience_input, fovea_y, fovea_x)

        prev_action = tf.argmax(a[..., :self.n_discrete_actions], axis=-1)[..., None]
        prev_action = tf.to_float(prev_action)

        return self._build_return_values(
            [fovea_x, fovea_y, prev_action, salience, glimpse, salience_input, _y], actions)

    def get_output(self, registers, actions):
        return actions[..., self.n_discrete_actions:]
