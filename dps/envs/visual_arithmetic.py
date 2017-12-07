
import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import CompositeEnv, InternalEnv
from dps.envs.grid_arithmetic import render_rollouts
from dps.supervised import ClassificationEnv, IntegerRegressionEnv
from dps.vision.train import EMNIST_CONFIG, SALIENCE_CONFIG
from dps.datasets import VisualArithmeticDataset
from dps.utils.tf import LeNet, MLP, SalienceMap, extract_glimpse_numpy_like
from dps.utils import Param, Config, image_to_string
from dps.updater import DifferentiableUpdater
from dps.rl.policy import EpsilonSoftmax, Beta, ProductDist, Policy, SigmoidNormal, SigmoidBeta


def sl_build_env():
    train = VisualArithmeticDataset(n_examples=cfg.n_train, one_hot=True)
    val = VisualArithmeticDataset(n_examples=cfg.n_val, one_hot=True)
    test = VisualArithmeticDataset(n_examples=cfg.n_val, one_hot=True)
    return ClassificationEnv(train, val, test, one_hot=True)


def sl_get_updater(env):
    build_model = LeNet(n_units=int(cfg.n_controller_units))
    return DifferentiableUpdater(env, build_model)


def build_env():
    internal = VisualArithmetic()

    train = VisualArithmeticDataset(n_examples=cfg.n_train, one_hot=False, image_shape=cfg.env_shape)
    for i in range(10):
        print(image_to_string(train.x[i, ...]))
    val = VisualArithmeticDataset(n_examples=cfg.n_val, one_hot=False, image_shape=cfg.env_shape)
    test = VisualArithmeticDataset(n_examples=cfg.n_val, one_hot=False, image_shape=cfg.env_shape)

    external = IntegerRegressionEnv(train, val, test)

    env = CompositeEnv(external, internal)
    env.obs_is_image = True
    return env


def build_policy(env, **kwargs):
    action_selection = ProductDist(
        EpsilonSoftmax(env.actions_dim-2, one_hot=True),
        # Beta(),
        # Beta()
        SigmoidBeta(c0_bounds=(1, 100), c1_bounds=(1, 100)),
        SigmoidBeta(c0_bounds=(1, 100), c1_bounds=(1, 100)),
        # SigmoidNormal(0.1),
        # SigmoidNormal(0.1)
        # SigmoidNormal(),
        # SigmoidNormal()
    )
    return Policy(action_selection, env.obs_shape, **kwargs)


config = Config(
    log_name='visual_arithmetic',
    render_rollouts=render_rollouts,
    build_env=build_env,
    build_policy=build_policy,

    reductions="sum",
    arithmetic_actions="+,*,max,min,+1",

    curriculum=[
        dict(env_shape=(28, 28), draw_shape=(14, 14), draw_offset=(7, 7)),
        dict(env_shape=(28, 28), draw_shape=(20, 20), draw_offset=(4, 4)),
        dict(env_shape=(28, 28), draw_shape=(28, 28), draw_offset=(0, 0)),
    ],
    stopping_criteria_name="01_loss",
    maximize_sc=False,
    threshold=0.1,
    base=10,
    T=30,
    min_digits=1,
    max_digits=1,
    final_reward=True,
    parity='both',
    max_overlap=10,

    start_loc=(0.5, 0.5),  # With respect to env_shape
    env_shape=(15, 15),
    draw_offset=(0, 0),
    draw_shape=(15, 15),
    sub_image_shape=(14, 14),

    n_train=10000,
    n_val=100,
    use_gpu=False,

    show_op=True,
    reward_window=0.4999,
    salience_action=True,
    salience_input_shape=(3*14, 3*14),
    salience_output_shape=(14, 14),
    initial_salience=False,
    visible_glimpse=False,

    ablation='easy',

    build_digit_classifier=lambda: LeNet(128, scope="digit_classifier"),
    build_op_classifier=lambda: LeNet(128, scope="op_classifier"),

    emnist_config=EMNIST_CONFIG.copy(),
    salience_config=SALIENCE_CONFIG.copy(
        min_digits=0,
        max_digits=4,
        std=0.05,
        n_units=100
    ),

    largest_digit=1000,

    n_glimpse_features=128,
)


def classifier_head(x):
    base = int(x.shape[-1])
    x = tf.stop_gradient(x)
    x = tf.argmax(x, -1)[..., None]
    x = tf.where(tf.equal(x, base), -1*tf.ones_like(x), x)
    x = tf.to_float(x)
    return x


class VisualArithmetic(InternalEnv):
    _action_names = ['classify_digit', 'classify_op', 'update_salience']

    @property
    def input_shape(self):
        return self.env_shape

    @property
    def n_discrete_actions(self):
        return len(self.action_names)

    arithmetic_actions = Param()
    base = Param()
    start_loc = Param()

    env_shape = Param()
    sub_image_shape = Param()

    visible_glimpse = Param()
    salience_action = Param()
    salience_input_shape = Param()
    salience_output_shape = Param()
    initial_salience = Param()

    op_classes = [chr(i + ord('A')) for i in range(26)]

    arithmetic_actions_dict = {
        '+': lambda acc, digit: acc + digit,
        '-': lambda acc, digit: acc - digit,
        '*': lambda acc, digit: acc * digit,
        '/': lambda acc, digit: acc / digit,
        'max': lambda acc, digit: tf.maximum(acc, digit),
        'min': lambda acc, digit: tf.minimum(acc, digit),
        '+1': lambda acc, digit: acc + 1,
        '-1': lambda acc, digit: acc - 1,
    }

    def __init__(self, **kwargs):
        self.sub_image_size = np.product(self.sub_image_shape)
        self.salience_input_size = np.product(self.salience_input_shape)
        self.salience_output_size = np.product(self.salience_output_shape)

        _arithmetic_actions = {}
        delim = ',' if ',' in self.arithmetic_actions else ' '
        for key in self.arithmetic_actions.split(delim):
            _arithmetic_actions[key] = self.arithmetic_actions_dict[key]
        self.arithmetic_actions = _arithmetic_actions

        self.action_names = (
            self._action_names +
            sorted(self.arithmetic_actions.keys()) +
            'fovea_x fovea_y'.split()
        )

        self.actions_dim = len(self.action_names)
        self.action_sizes = [1] * self.actions_dim

        self._init_networks()
        self._init_rb()

        super(VisualArithmetic, self).__init__()

    def _init_rb(self):
        values = (
            [0., 0., -1., .5, .5, -1.] +
            [np.zeros(self.salience_output_size, dtype='f')] +
            [np.zeros(self.sub_image_size, dtype='f')] +
            [np.zeros(self.salience_input_size, dtype='f')]
        )

        if self.visible_glimpse:
            self.rb = RegisterBank(
                'VisualArithmeticRB',
                'digit op acc fovea_x fovea_y prev_action salience glimpse',
                'salience_input', values=values, output_names='acc',
                no_display='glimpse salience salience_input',
            )
        else:
            self.rb = RegisterBank(
                'VisualArithmeticRB',
                'digit op acc fovea_x fovea_y prev_action salience',
                'glimpse salience_input', values=values, output_names='acc',
                no_display='glimpse salience salience_input',
            )

    def _init_networks(self):
        digit_config = cfg.emnist_config.copy(
            classes=list(range(self.base)),
            build_function=cfg.build_digit_classifier
        )

        self.digit_classifier = cfg.build_digit_classifier()
        self.digit_classifier.set_pretraining_params(
            digit_config, name_params='classes include_blank shape n_controller_units',
            directory=cfg.model_dir + '/emnist_pretrained'
        )

        op_config = cfg.emnist_config.copy(
            classes=list(self.op_classes),
            build_function=cfg.build_op_classifier
        )

        self.op_classifier = cfg.build_op_classifier()
        self.op_classifier.set_pretraining_params(
            op_config, name_params='classes include_blank shape n_controller_units',
            directory=cfg.model_dir + '/emnist_pretrained',
        )

        self.classifier_head = classifier_head

        self.maybe_build_salience_detector()

    def maybe_build_salience_detector(self):
        if self.salience_action:
            def _build_salience_detector(output_shape=self.salience_output_shape):
                return SalienceMap(
                    2 * cfg.max_digits,
                    MLP([cfg.n_units, cfg.n_units, cfg.n_units], scope="salience_detector"),
                    output_shape, std=cfg.std, flatten_output=True
                )

            salience_config = cfg.salience_config.copy(
                output_shape=self.salience_output_shape,
                image_shape=self.salience_input_shape,
                build_function=_build_salience_detector,
            )

            with salience_config:
                self.salience_detector = _build_salience_detector()

            self.salience_detector.set_pretraining_params(
                salience_config,
                name_params='classes std min_digits max_digits n_units '
                            'sub_image_shape image_shape output_shape',
                directory=cfg.model_dir + '/salience_pretrained'
            )
        else:
            self.salience_detector = None

    def _build_update_glimpse(self, fovea_y, fovea_x):
        fovea_center_px = tf.concat([fovea_y, fovea_x], axis=-1) * self.env_shape

        fovea_top_left_px = fovea_center_px - 0.5 * np.array(self.sub_image_shape)

        inp = self.input_ph[..., None]

        glimpse = extract_glimpse_numpy_like(
            inp, self.sub_image_shape, fovea_top_left_px, fill_value=0.0)
        glimpse = tf.reshape(glimpse, (-1, self.sub_image_size), name="glimpse")
        return glimpse

    def _build_update_salience(self, update_salience, salience, salience_input, fovea_y, fovea_x):
        fovea_center_px = tf.concat([fovea_y, fovea_x], axis=-1) * self.env_shape

        salience_input_top_left_px = fovea_center_px - np.array(self.salience_input_shape) / 2.0

        inp = self.input_ph[..., None]

        glimpse = extract_glimpse_numpy_like(
            inp, self.salience_input_shape, salience_input_top_left_px, fill_value=0.0)

        new_salience = self.salience_detector(glimpse, self.salience_output_shape, False)
        new_salience = tf.reshape(new_salience, (-1, self.salience_output_size))

        new_salience_input = tf.reshape(glimpse, (-1, self.salience_input_size))

        salience = (1 - update_salience) * salience + update_salience * new_salience
        salience_input = (1 - update_salience) * salience_input + update_salience * new_salience_input

        return salience, salience_input

    def _build_update_storage(self, glimpse, prev_digit, classify_digit, prev_op, classify_op):
        digit = self.classifier_head(self.digit_classifier(glimpse, self.base + 1, False))
        new_digit = (1 - classify_digit) * prev_digit + classify_digit * digit

        op = self.classifier_head(self.op_classifier(glimpse, len(self.op_classes) + 1, False))
        new_op = (1 - classify_op) * prev_op + classify_op * op

        return new_digit, new_op

    def _build_return_values(self, registers, actions):
        new_registers = self.rb.wrap(*registers)
        reward = self.build_reward(new_registers, actions)
        done = tf.zeros(tf.shape(new_registers)[:-1])[..., None]
        return done, reward, new_registers

    def build_init(self, r):
        self.maybe_build_placeholders()

        (_digit, _op, _acc, _fovea_x, _fovea_y, _prev_action,
            _salience, _glimpse, _salience_input) = self.rb.as_tuple(r)
        batch_size = tf.shape(self.input_ph)[0]

        # init fovea
        if self.start_loc is not None:
            fovea_y = tf.fill((batch_size, 1), self.start_loc[0])
            fovea_x = tf.fill((batch_size, 1), self.start_loc[1])
        else:
            fovea_y = tf.random_uniform(
                tf.shape(fovea_y), 0, 1, dtype=tf.int32)
            fovea_x = tf.random_uniform(
                tf.shape(fovea_x), 0, 1, dtype=tf.int32)

        fovea_y = tf.cast(fovea_y, tf.float32)
        fovea_x = tf.cast(fovea_x, tf.float32)

        glimpse = self._build_update_glimpse(fovea_y, fovea_x)

        salience = _salience
        salience_input = _salience_input
        if self.initial_salience:
            salience, salience_input = self._build_update_salience(
                1.0, _salience, _salience_input, _fovea_y, _fovea_x)

        digit = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        op = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        acc = -1 * tf.ones((batch_size, 1), dtype=tf.float32)

        return self.rb.wrap(digit, op, acc, fovea_x, fovea_y, _prev_action, salience, glimpse, salience_input)

    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        (classify_digit, classify_op, update_salience, *arithmetic_actions, new_fovea_x, new_fovea_y) = actions

        salience = _salience
        salience_input = _salience_input
        if self.salience_action:
            salience, salience_input = self._build_update_salience(
                update_salience, _salience, _salience_input, _fovea_y, _fovea_x)

        digit = tf.zeros_like(_digit)
        acc = tf.zeros_like(_acc)

        original_factor = tf.ones_like(classify_digit)
        for key, action in zip(sorted(self.arithmetic_actions), arithmetic_actions):
            original_factor -= action
            acc += action * self.arithmetic_actions[key](_acc, _digit)
        acc += original_factor * _acc

        acc = tf.clip_by_value(acc, -1000.0, 1000.0)

        digit, op = self._build_update_storage(_glimpse, _digit, classify_digit, _op, classify_op)
        glimpse = self._build_update_glimpse(new_fovea_y, new_fovea_x)

        prev_action = tf.argmax(a, axis=-1)[..., None]
        prev_action = tf.to_float(prev_action)

        return self._build_return_values(
            [digit, op, acc, new_fovea_x, new_fovea_y, prev_action, salience, glimpse, salience_input],
            actions)


class VisualArithmeticEasy(VisualArithmetic):
    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        classify_digit, classify_op, update_salience, *arithmetic_actions, new_fovea_x, new_fovea_y = actions

        salience = _salience
        salience_input = _salience_input
        if self.salience_action:
            salience, salience_input = self._build_update_salience(
                update_salience, _salience, _salience_input, _fovea_y, _fovea_x)

        op = self.classifier_head(self.op_classifier(_glimpse, len(self.op_classes) + 1, False))
        op = (1 - classify_op) * _op + classify_op * op

        new_digit_factor = classify_digit
        for action in arithmetic_actions:
            new_digit_factor += action

        digit = self.classifier_head(self.digit_classifier(_glimpse, self.base + 1, False))
        digit = (1 - new_digit_factor) * _digit + new_digit_factor * digit

        new_acc_factor = tf.zeros_like(classify_digit)
        acc = tf.zeros_like(_acc)
        for key, action in zip(sorted(self.arithmetic_actions), arithmetic_actions):
            new_acc_factor += action
            # Its crucial that we use `digit` here and not `_digit`
            acc += action * self.arithmetic_actions[key](_acc, digit)
        acc += (1 - new_acc_factor) * _acc

        acc = tf.clip_by_value(acc, -1000.0, 1000.0)

        glimpse = self._build_update_glimpse(new_fovea_y, new_fovea_x)

        prev_action = tf.argmax(a, axis=-1)[..., None]
        prev_action = tf.to_float(prev_action)

        return self._build_return_values(
            [digit, op, acc, new_fovea_x, new_fovea_y, prev_action, salience, glimpse, salience_input],
            actions)
