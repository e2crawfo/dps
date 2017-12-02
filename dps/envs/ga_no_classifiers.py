import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.supervised import IntegerRegressionEnv
from dps.environment import CompositeEnv
from dps.utils.tf import LeNet, MLP, CompositeCell
from dps.utils import image_to_string, Config
from dps.rl.policy import EpsilonSoftmax, ProductDist, Policy

from dps.envs.grid_arithmetic import GridArithmeticDataset, GridArithmetic
from dps.envs.grid_arithmetic import config as ga_config


def build_env():
    internal = GridArithmeticNoClassifiers()

    train = GridArithmeticDataset(n_examples=cfg.n_train, one_hot=False)
    val = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=False)
    test = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=False)
    external = IntegerRegressionEnv(train, val, test)

    env = CompositeEnv(external, internal)
    env.obs_is_image = True
    return env


def build_policy(env, **kwargs):
    action_selection = ProductDist(
        EpsilonSoftmax(env.n_discrete_actions, one_hot=True),
        EpsilonSoftmax(10, one_hot=False))
    return Policy(action_selection, env.obs_shape, **kwargs)


def no_classifiers_inp(obs):
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
        MLP(), params_dim, inp=no_classifiers_inp, name=name)


def ga_no_classifiers_render_rollouts(env, rollouts):
    registers = np.concatenate([rollouts.obs, rollouts.hidden], axis=2)
    registers = np.concatenate(
        [registers, rollouts._metadata['final_registers'][np.newaxis, ...]],
        axis=0)

    internal = env.internal

    for i in range(registers.shape[1]):
        glimpse = internal.rb.get("glimpse", registers[:, i, :])
        glimpse = glimpse.reshape((glimpse.shape[0],) + internal.image_shape)

        salience_input = internal.rb.get("salience_input", registers[:, i, :])
        salience_input = salience_input.reshape(
            (salience_input.shape[0],) + internal.salience_input_shape)

        salience = internal.rb.get("salience", registers[:, i, :])
        salience = salience.reshape(
            (salience.shape[0],) + internal.salience_output_shape)

        prev_digit = internal.rb.get("prev_digit", registers[:, i, :])
        acc = internal.rb.get("acc", registers[:, i, :])

        actions = rollouts.a[:, i, :]

        print("Start of rollout {}.".format(i))
        for t in range(rollouts.T):
            print("t={}".format(t) + " * " * 20)
            action_idx = int(np.argmax(actions[t, :env.n_discrete_actions]))

            print("prev_digit: ", prev_digit[t])
            print("acc: ", acc[t])

            print(image_to_string(glimpse[t]))
            print("\n")
            print(image_to_string(salience_input[t]))
            print("\n")
            print(image_to_string(salience[t]))
            print("\n")

            print("\ndiscrete action={}".format(internal.action_names[action_idx]))
            print("\nother action={}".format(actions[t, env.n_discrete_actions:]))


config_delta = Config(
    log_name='grid_arithmetic_no_classifiers',
    render_rollouts=ga_no_classifiers_render_rollouts,
    build_env=build_env,
    build_policy=build_policy,
    build_controller=build_controller,
    n_glimpse_features=128,
    n_glimpse_units=128,
)


config = ga_config.copy()
config.update(config_delta)


class GridArithmeticNoClassifiers(GridArithmetic):
    has_differentiable_loss = True
    _action_names = ['>', '<', 'v', '^', 'update_salience']

    @property
    def n_discrete_actions(self):
        return 5 + len(self.arithmetic_actions)

    def __init__(self, **kwargs):
        super(GridArithmeticNoClassifiers, self).__init__()

        self.action_names.append('digit')
        self.action_sizes.append(1)
        self.actions_dim = sum(self.action_sizes)

    def build_init(self, r):
        self.maybe_build_placeholders()

        _prev_digit, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

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

        prev_digit = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        acc = -1 * tf.ones((batch_size, 1), dtype=tf.float32)

        return self.rb.wrap(prev_digit, acc, fovea_x, fovea_y, _prev_action, salience, glimpse, salience_input)

    def _init_networks(self):
        self.maybe_build_salience_detector()

    def _init_rb(self):
        values = (
            [-1., -1., 0., 0., -1.] +
            [np.zeros(self.salience_output_size, dtype='f')] +
            [np.zeros(self.image_size, dtype='f')] +
            [np.zeros(self.salience_input_size, dtype='f')]
        )

        self.rb = RegisterBank(
            'GridArithmeticNoClassifiersRB',
            'prev_digit acc fovea_x fovea_y prev_action salience glimpse', 'salience_input', values=values,
            no_display='glimpse salience salience_input', output_names='acc',
        )

    def build_step(self, t, r, a):
        _prev_digit, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        (right, left, down, up, update_salience, *arithmetic_actions, digit) = actions

        salience = _salience
        salience_input = _salience_input
        if self.salience_action:
            salience, salience_input = self._build_update_salience(
                update_salience, _salience, _salience_input, _fovea_y, _fovea_x)

        acc = tf.zeros_like(_acc)

        original_factor = tf.ones_like(right)
        for key, action in zip(sorted(self.arithmetic_actions), arithmetic_actions):
            original_factor -= action
            acc += action * self.arithmetic_actions[key](_acc, digit)
        acc += original_factor * _acc

        acc = tf.clip_by_value(acc, -1000.0, 1000.0)

        fovea_y, fovea_x = self._build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)
        glimpse = self._build_update_glimpse(fovea_y, fovea_x)

        prev_action = tf.cast(tf.reshape(tf.argmax(a, axis=1), (-1, 1)), tf.float32)

        return self._build_return_values(
            [digit, acc, fovea_x, fovea_y, prev_action, salience, glimpse, salience_input], actions)
