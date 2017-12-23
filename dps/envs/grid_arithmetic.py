import tensorflow as tf
import numpy as np
import os
from collections import OrderedDict

from dps import cfg
from dps.register import RegisterBank
from dps.environment import CompositeEnv, InternalEnv
from dps.supervised import ClassificationEnv, IntegerRegressionEnv
from dps.vision import EMNIST_CONFIG, SALIENCE_CONFIG, OMNIGLOT_CONFIG
from dps.datasets import GridArithmeticDataset, GridOmniglotDataset, OmniglotDataset
from dps.utils.tf import LeNet, MLP, SalienceMap, extract_glimpse_numpy_like, ScopedFunction
from dps.utils import Param, Config, image_to_string
from dps.updater import DifferentiableUpdater
from dps.rl.policy import EpsilonSoftmax, DiscretePolicy
from dps.envs.visual_arithmetic import digit_map, classifier_head, ResizeSalienceDetector


def sl_build_env():
    digits = digit_map[cfg.parity]
    train = GridArithmeticDataset(n_examples=cfg.n_train, one_hot=True, digits=digits)
    val = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=True, digits=digits)
    test = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=True, digits=digits)
    return ClassificationEnv(train, val, test, one_hot=True)


def sl_build_model():
    if cfg.mode == "standard":
        return cfg.build_convolutional_model()
    elif cfg.mode == "pretrained":
        return FeedforwardPretrained()
    else:
        raise Exception()


def sl_get_updater(env):
    model = cfg.build_model()

    return DifferentiableUpdater(env, model)


class FeedforwardPretrained(ScopedFunction):
    fixed = Param()
    pretrain = Param()
    emnist_config = Param()
    build_digit_classifier = Param()
    build_op_classifier = Param()
    build_feedforward_model = Param()

    include_raw = Param()
    n_raw_features = Param()
    build_convolutional_model = Param()

    image_shape_grid = Param()
    sub_image_shape = Param()

    def __init__(self, **kwargs):
        super(FeedforwardPretrained, self).__init__(**kwargs)

        self.digit_classifier = None
        self.op_classifier = None
        self.feedforward_model = None

    def _call(self, inp, output_size, is_training):
        if self.digit_classifier is None:

            self.digit_classifier = self.build_digit_classifier()

            if self.pretrain:
                digit_config = self.emnist_config.copy(
                    classes=list(range(10)),
                    build_function=self.build_digit_classifier,
                    include_blank=True
                )
                self.digit_classifier.set_pretraining_params(
                    digit_config, name_params='classes include_blank shape n_controller_units',
                    directory=cfg.model_dir + '/emnist_pretrained'
                )

            if self.fixed:
                self.digit_classifier.fix_variables()

        if self.op_classifier is None:

            self.op_classifier = self.build_op_classifier()

            if self.pretrain:
                op_config = self.emnist_config.copy(
                    classes=list(GridArithmetic.op_classes),
                    build_function=self.build_op_classifier
                )

                self.op_classifier.set_pretraining_params(
                    op_config, name_params='classes include_blank shape n_controller_units',
                    directory=cfg.model_dir + '/emnist_pretrained'
                )

            if self.fixed:
                self.op_classifier.fix_variables()

        def head(inp):
            if self.fixed:
                return tf.stop_gradient(inp)
            else:
                return inp

        # Assume inp is of shape (batch_size, h, w, d)
        rows = tf.split(inp, self.image_shape_grid[0], axis=1)
        _rows = [tf.split(row, self.image_shape_grid[1], axis=2) for row in rows]

        digit_classifications, op_classifications = [], []
        for row in _rows:
            digit_classifications.append(
                [head(self.digit_classifier(img, 11, is_training)) for img in row])
            op_classifications.append(
                [head(self.op_classifier(img, len(GridArithmetic.op_classes) + 1, is_training)) for img in row])

        if self.feedforward_model is None:
            self.feedforward_model = cfg.build_feedforward_model()

        tensors = (
            [c for row in digit_classifications for c in row] +
            [c for row in op_classifications for c in row])
        model_input = tf.concat(tensors, axis=-1, name="feedforward_pretrained_output_flattened")

        if self.include_raw:
            self.convolutional_model = self.build_convolutional_model()
            raw_features = self.convolutional_model(inp, self.n_raw_features, is_training)
            model_input = tf.concat([model_input, raw_features], axis=-1)

        return self.feedforward_model(model_input, output_size, is_training)


def build_env():
    if cfg.ablation == 'omniglot':
        if not cfg.omniglot_classes:
            cfg.omniglot_classes = OmniglotDataset.sample_classes(10)
        internal = OmniglotCounting()
    elif cfg.ablation == 'easy':
        internal = GridArithmeticEasy()
    else:
        internal = GridArithmetic()

    if cfg.ablation == 'omniglot':
        with Config(classes=cfg.omniglot_classes, target_loc=cfg.op_loc):
            train = GridOmniglotDataset(n_examples=cfg.n_train, indices=range(15))
            val = GridOmniglotDataset(n_examples=cfg.n_val, indices=range(15))
            test = GridOmniglotDataset(n_examples=cfg.n_val, indices=range(15, 20))
    else:
        digits = digit_map[cfg.parity]
        train = GridArithmeticDataset(n_examples=cfg.n_train, one_hot=False, digits=digits)
        val = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=False, digits=digits)
        test = GridArithmeticDataset(n_examples=cfg.n_val, one_hot=False, digits=digits)

    external = IntegerRegressionEnv(train, val, test)

    env = CompositeEnv(external, internal)
    env.obs_is_image = True
    return env


def build_policy(env, **kwargs):
    action_selection = EpsilonSoftmax(env.actions_dim, one_hot=True)
    return DiscretePolicy(action_selection, env.obs_shape, **kwargs)


class _GridArithmeticRenderRollouts(object):
    image_fields = "glimpse salience_input salience".split()

    def _render_rollout(self, env, T, action_names, actions, fields, f):
        if 'glimpse' in fields:
            glimpse = fields['glimpse']
            glimpse = glimpse.reshape((glimpse.shape[0],) + env.internal.sub_image_shape)
        else:
            glimpse = None

        if 'salience_input' in fields:
            salience_input = fields['salience_input']
            salience_input = salience_input.reshape(
                (salience_input.shape[0],) + env.internal.salience_input_shape)
        else:
            salience_input = None

        if 'salience' in fields:
            salience = fields['salience']
            salience = salience.reshape(
                (salience.shape[0],) + env.internal.salience_output_shape)
        else:
            salience = None

        for t in range(T):
            print("t={}".format(t) + "* " * 20, file=f)

            if glimpse is not None:
                print('glimpse', file=f)
                print(image_to_string(glimpse[t]), file=f)
                print("\n", file=f)

            if salience_input is not None:
                print('salience_input', file=f)
                print(image_to_string(salience_input[t]), file=f)
                print("\n", file=f)

            if salience is not None:
                print('salience', file=f)
                print(image_to_string(salience[t]), file=f)
                print("\n", file=f)

            for k, v in fields.items():
                if k not in self.image_fields:
                    print("{}: {}".format(k, v[t]), file=f)

            action_idx = int(np.argmax(actions[t, :env.n_discrete_actions]))
            print("\ndiscrete action={}".format(action_names[action_idx]), file=f)
            print("\nother action={}".format(actions[t, env.n_discrete_actions:]), file=f)

    def __call__(self, env, rollouts):
        self.env = env
        registers = np.concatenate([rollouts.obs, rollouts.hidden], axis=2)
        registers = np.concatenate(
            [registers, rollouts._metadata['final_registers'][np.newaxis, ...]],
            axis=0)

        internal = env.internal

        path = os.path.join(cfg.path, 'rollouts')
        os.makedirs(path, exist_ok=True)

        for i in range(registers.shape[1]):
            fields = internal.rb.as_dict(registers[:, i, :])
            fields = OrderedDict((k, fields[k]) for k in sorted(fields.keys()))

            actions = rollouts.a[:, i, :]

            with open(os.path.join(path, str(i)), 'w') as f:
                print("Start of rollout {}.".format(i), file=f)
                self._render_rollout(env, rollouts.T, internal.action_names, actions, fields, f)


render_rollouts = _GridArithmeticRenderRollouts()


config = Config(
    log_name='grid_arithmetic',
    render_rollouts=render_rollouts,
    build_env=build_env,
    build_policy=build_policy,

    # Traing specific
    threshold=0.04,
    curriculum=[dict()],
    n_train=10000,
    n_val=100,
    use_gpu=False,

    # env-specific
    reductions="A:sum,M:prod,X:max,N:min",
    op_loc=(0, 0),  # With respect to draw_shape_grid
    start_loc=(0, 0),  # With respect to image_shape_grid
    image_shape_grid=(2, 2),
    draw_offset=(0, 0),
    draw_shape_grid=None,
    sub_image_shape=(14, 14),
    min_digits=2,
    max_digits=3,
    parity='both',
    largest_digit=1000,

    # RL-specific
    ablation='easy',
    arithmetic_actions="+,*,max,min,+1",
    reward_window=0.4999,
    salience_action=True,
    salience_input_shape=(3*14, 3*14),
    salience_output_shape=(14, 14),
    initial_salience=False,
    salience_model=True,
    visible_glimpse=False,
    final_reward=True,
    T=30,

    # Pre-trained modules
    build_digit_classifier=lambda: LeNet(128, scope="digit_classifier"),
    build_op_classifier=lambda: LeNet(128, scope="op_classifier"),
    build_omniglot_classifier=lambda: LeNet(128, scope="omniglot_classifier"),

    emnist_config=EMNIST_CONFIG.copy(),
    salience_config=SALIENCE_CONFIG.copy(
        min_digits=0,
        max_digits=4,
        std=0.05,
        n_units=100
    ),
    omniglot_config=OMNIGLOT_CONFIG.copy(),
    omniglot_classes=[
        'Cyrillic,17', 'Mkhedruli_(Georgian),5', 'Bengali,23', 'Mongolian,19',
        'Malayalam,3', 'Ge_ez,15', 'Glagolitic,33', 'Tagalog,11', 'Gujarati,23',
        'Old_Church_Slavonic_(Cyrillic),7'],  # Chosen randomly from set of all omniglot symbols.
)


feedforward_config = config.copy(
    get_updater=sl_get_updater,
    build_model=sl_build_model,
    build_env=sl_build_env,

    use_gpu=False,
    per_process_gpu_memory_fraction=0.2,

    largest_digit=99,
    batch_size=16,
    optimizer_spec="adam",
    opt_steps_per_update=1,
    lr_schedule="1e-4",

    mode="standard",
    build_convolutional_model=lambda: LeNet(cfg.n_controller_units),

    # For when mode == "pretrained"
    fixed=True,
    pretrain=True,
    build_feedforward_model=lambda: MLP(
        [cfg.n_controller_units, cfg.n_controller_units, cfg.n_controller_units]),
    include_raw=True,
    n_raw_features=128,
)


class GridArithmetic(InternalEnv):
    _action_names = ['>', '<', 'v', '^', 'classify_digit', 'classify_op', 'update_salience']

    @property
    def input_shape(self):
        return tuple(es*s for es, s in zip(self.env_shape, self.sub_image_shape))

    @property
    def n_discrete_actions(self):
        return len(self.action_names)

    arithmetic_actions = Param()
    env_shape = Param(aliases="image_shape_grid")
    start_loc = Param()
    sub_image_shape = Param()
    visible_glimpse = Param()
    salience_action = Param()
    salience_model = Param()
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
            sorted(self.arithmetic_actions.keys())
        )

        self.actions_dim = len(self.action_names)
        self.action_sizes = [1] * self.actions_dim
        self._init_networks()
        self._init_rb()

        super(GridArithmetic, self).__init__()

    def _init_rb(self):
        values = (
            [0., 0., -1., 0., 0., -1.] +
            [np.zeros(self.salience_output_size, dtype='f')] +
            [np.zeros(self.sub_image_size, dtype='f')] +
            [np.zeros(self.salience_input_size, dtype='f')]
        )

        if self.visible_glimpse:
            self.rb = RegisterBank(
                'GridArithmeticRB',
                'digit op acc fovea_x fovea_y prev_action salience glimpse', 'salience_input', values=values,
                output_names='acc', no_display='glimpse salience salience_input',
            )
        else:
            self.rb = RegisterBank(
                'GridArithmeticRB',
                'digit op acc fovea_x fovea_y prev_action salience', 'glimpse salience_input', values=values,
                output_names='acc', no_display='glimpse salience salience_input',
            )

    def _init_networks(self):
        digit_config = cfg.emnist_config.copy(
            classes=list(range(10)),
            build_function=cfg.build_digit_classifier,
            include_blank=True
        )

        self.digit_classifier = cfg.build_digit_classifier()
        self.digit_classifier.set_pretraining_params(
            digit_config, name_params='classes include_blank shape n_controller_units',
            directory=cfg.model_dir + '/emnist_pretrained'
        )
        self.digit_classifier.fix_variables()

        op_config = cfg.emnist_config.copy(
            classes=list(self.op_classes),
            build_function=cfg.build_op_classifier
        )

        self.op_classifier = cfg.build_op_classifier()
        self.op_classifier.set_pretraining_params(
            op_config, name_params='classes include_blank shape n_controller_units',
            directory=cfg.model_dir + '/emnist_pretrained',
        )
        self.op_classifier.fix_variables()

        self.classifier_head = classifier_head

        self.maybe_build_salience_detector()

    def maybe_build_salience_detector(self):
        if self.salience_action:
            if self.salience_model:
                def _build_salience_detector(output_shape=self.salience_output_shape):
                    return SalienceMap(
                        2 * cfg.max_digits,
                        MLP([cfg.n_units, cfg.n_units, cfg.n_units], scope="salience_detector"),
                        output_shape, std=cfg.std, flatten_output=True
                    )

                salience_config = cfg.salience_config.copy(
                    output_shape=self.salience_output_shape,
                    image_shape=self.salience_input_shape,
                    sub_image_shape=self.sub_image_shape,
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
                self.salience_detector.fix_variables()
            else:
                self.salience_detector = ResizeSalienceDetector()
        else:
            self.salience_detector = None

    def _build_update_glimpse(self, fovea_y, fovea_x):
        top_left = tf.concat([fovea_y, fovea_x], axis=-1) * self.sub_image_shape

        inp = self.input_ph[..., None]

        glimpse = extract_glimpse_numpy_like(
            inp, self.sub_image_shape, top_left, fill_value=0.0)
        glimpse = tf.reshape(glimpse, (-1, self.sub_image_size), name="glimpse")
        return glimpse

    def _build_update_salience(self, update_salience, salience, salience_input, fovea_y, fovea_x):
        top_left = tf.concat([fovea_y, fovea_x], axis=-1) * self.sub_image_shape
        top_left -= (np.array(self.salience_input_shape) - np.array(self.sub_image_shape)) / 2.0

        inp = self.input_ph[..., None]

        glimpse = extract_glimpse_numpy_like(
            inp, self.salience_input_shape, top_left, fill_value=0.0)

        new_salience = tf.stop_gradient(self.salience_detector(glimpse, self.salience_output_shape, False))
        new_salience = tf.reshape(new_salience, (-1, self.salience_output_size))

        new_salience_input = tf.reshape(glimpse, (-1, self.salience_input_size))

        salience = (1 - update_salience) * salience + update_salience * new_salience
        salience_input = (1 - update_salience) * salience_input + update_salience * new_salience_input
        return salience, salience_input

    def _build_update_storage(self, glimpse, prev_digit, classify_digit, prev_op, classify_op):
        digit = self.classifier_head(self.digit_classifier(glimpse, 11, False))
        new_digit = (1 - classify_digit) * prev_digit + classify_digit * digit

        op = self.classifier_head(self.op_classifier(glimpse, len(self.op_classes) + 1, False))
        new_op = (1 - classify_op) * prev_op + classify_op * op

        return new_digit, new_op

    def _build_update_fovea(self, right, left, down, up, fovea_y, fovea_x):
        fovea_x = (1 - right - left) * fovea_x + \
            right * (fovea_x + 1) + \
            left * (fovea_x - 1)
        fovea_y = (1 - down - up) * fovea_y + \
            down * (fovea_y + 1) + \
            up * (fovea_y - 1)
        fovea_y = tf.clip_by_value(fovea_y, 0, self.env_shape[0]-1)
        fovea_x = tf.clip_by_value(fovea_x, 0, self.env_shape[1]-1)
        return fovea_y, fovea_x

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

        digit = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        op = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        acc = -1 * tf.ones((batch_size, 1), dtype=tf.float32)

        return self.rb.wrap(
            digit, op, acc, fovea_x, fovea_y,
            _prev_action, salience, glimpse, salience_input)

    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        (right, left, down, up, classify_digit, classify_op,
            update_salience, *arithmetic_actions) = actions

        salience = _salience
        salience_input = _salience_input
        if self.salience_action:
            salience, salience_input = self._build_update_salience(
                update_salience, _salience, _salience_input, _fovea_y, _fovea_x)

        digit = tf.zeros_like(_digit)
        acc = tf.zeros_like(_acc)

        original_factor = tf.ones_like(right)
        for key, action in zip(sorted(self.arithmetic_actions), arithmetic_actions):
            original_factor -= action
            acc += action * self.arithmetic_actions[key](_acc, _digit)
        acc += original_factor * _acc

        acc = tf.clip_by_value(acc, -1000.0, 1000.0)

        digit, op = self._build_update_storage(_glimpse, _digit, classify_digit, _op, classify_op)
        fovea_y, fovea_x = self._build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)
        glimpse = self._build_update_glimpse(fovea_y, fovea_x)

        prev_action = tf.argmax(a, axis=-1)[..., None]
        prev_action = tf.to_float(prev_action)

        return self._build_return_values(
            [digit, op, acc, fovea_x, fovea_y, prev_action, salience, glimpse, salience_input],
            actions)


class GridArithmeticEasy(GridArithmetic):
    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        (right, left, down, up, classify_digit, classify_op,
            update_salience, *arithmetic_actions) = actions

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

        digit = self.classifier_head(self.digit_classifier(_glimpse, 11, False))
        digit = (1 - new_digit_factor) * _digit + new_digit_factor * digit

        new_acc_factor = tf.zeros_like(right)
        acc = tf.zeros_like(_acc)
        for key, action in zip(sorted(self.arithmetic_actions), arithmetic_actions):
            new_acc_factor += action
            # Its crucial that we use `digit` here and not `_digit`
            acc += action * self.arithmetic_actions[key](_acc, digit)
        acc += (1 - new_acc_factor) * _acc

        acc = tf.clip_by_value(acc, -1000.0, 1000.0)

        fovea_y, fovea_x = self._build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)
        glimpse = self._build_update_glimpse(fovea_y, fovea_x)

        prev_action = tf.argmax(a, axis=-1)[..., None]
        prev_action = tf.to_float(prev_action)

        return self._build_return_values(
            [digit, op, acc, fovea_x, fovea_y, prev_action, salience, glimpse, salience_input],
            actions)


class OmniglotCounting(GridArithmeticEasy):
    _action_names = GridArithmeticEasy._action_names + ['classify_omniglot']

    omniglot_classes = Param()

    def _init_rb(self):
        values = (
            [0., 0., 0., -1., 0., 0., -1.] +
            [np.zeros(self.salience_output_size, dtype='f')] +
            [np.zeros(self.sub_image_size, dtype='f')] +
            [np.zeros(self.salience_input_size, dtype='f')]
        )

        if self.visible_glimpse:
            self.rb = RegisterBank(
                'GridArithmeticRB',
                'omniglot digit op acc fovea_x fovea_y prev_action salience glimpse', 'salience_input', values=values,
                output_names='acc', no_display='glimpse salience salience_input',
            )
        else:
            self.rb = RegisterBank(
                'GridArithmeticRB',
                'omniglot digit op acc fovea_x fovea_y prev_action salience', 'glimpse salience_input', values=values,
                output_names='acc', no_display='glimpse salience salience_input',
            )

    def build_step(self, t, r, a):
        _omniglot, _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        (right, left, down, up, classify_digit, classify_op, classify_omniglot,
            update_salience, *arithmetic_actions) = actions

        salience = _salience
        salience_input = _salience_input
        if self.salience_action:
            salience, salience_input = self._build_update_salience(
                update_salience, _salience, _salience_input, _fovea_y, _fovea_x)

        omniglot = self.classifier_head(self.omniglot_classifier(_glimpse, len(self.omniglot_classes) + 1, False))
        omniglot = (1 - classify_omniglot) * _omniglot + classify_omniglot * omniglot

        op = self.classifier_head(self.op_classifier(_glimpse, len(self.op_classes) + 1, False))
        op = (1 - classify_op) * _op + classify_op * op

        new_digit_factor = classify_digit
        for action in arithmetic_actions:
            new_digit_factor += action

        digit = self.classifier_head(self.digit_classifier(_glimpse, 11, False))
        digit = (1 - new_digit_factor) * _digit + new_digit_factor * digit

        new_acc_factor = tf.zeros_like(right)
        acc = tf.zeros_like(_acc)
        for key, action in zip(sorted(self.arithmetic_actions), arithmetic_actions):
            new_acc_factor += action
            # Its crucial that we use `digit` here and not `_digit`
            acc += action * self.arithmetic_actions[key](_acc, digit)
        acc += (1 - new_acc_factor) * _acc

        acc = tf.clip_by_value(acc, -1000.0, 1000.0)

        fovea_y, fovea_x = self._build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)
        glimpse = self._build_update_glimpse(fovea_y, fovea_x)

        prev_action = tf.argmax(a, axis=-1)[..., None]
        prev_action = tf.to_float(prev_action)

        return self._build_return_values(
            [omniglot, digit, op, acc, fovea_x, fovea_y, prev_action, salience, glimpse, salience_input], actions)

    def _init_networks(self):
        super(OmniglotCounting, self)._init_networks()
        omniglot_config = cfg.omniglot_config.copy(
            classes=self.omniglot_classes,
            build_function=cfg.build_omniglot_classifier,
        )

        self.omniglot_classifier = cfg.build_omniglot_classifier()
        self.omniglot_classifier.set_pretraining_params(
            omniglot_config,
            name_params='classes include_blank shape n_controller_units',
            directory=cfg.model_dir + '/omniglot_pretrained',
        )
        self.omniglot_classifier.fix_variables()

    def build_init(self, r):
        self.maybe_build_placeholders()

        (_omniglot, _digit, _op, _acc, _fovea_x, _fovea_y,
            _prev_action, _salience, _glimpse, _salience_input) = self.rb.as_tuple(r)

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
                1.0, _salience, _salience_input, fovea_y, fovea_x)

        omniglot = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        digit = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        op = -1 * tf.ones((batch_size, 1), dtype=tf.float32)
        acc = -1 * tf.ones((batch_size, 1), dtype=tf.float32)

        return self.rb.wrap(
            omniglot, digit, op, acc, fovea_x, fovea_y,
            _prev_action, salience, glimpse, salience_input)
