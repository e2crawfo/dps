import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import (
    RegressionDataset, RegressionEnv, CompositeEnv, InternalEnv)
from dps.vision import MNIST_CONFIG, MNIST_SALIENCE_CONFIG
from dps.utils.tf import LeNet, MLP, SalienceMap, extract_glimpse_numpy_like, resize_image_with_crop_or_pad
from dps.utils import DataContainer, Param, Config, image_to_string
from dps.rl.policy import Softmax, Normal, ProductDist, Policy, DiscretePolicy
from dps.test.test_mnist import salience_render_hook

from mnist_arithmetic import load_emnist


def grid_arithmetic_render_rollouts(env, rollouts):
    registers = np.concatenate([rollouts.obs, rollouts.hidden], axis=2)
    registers = np.concatenate(
        [registers, rollouts._metadata['final_registers'][np.newaxis, ...]],
        axis=0)

    internal = env.internal

    for i in range(registers.shape[1]):
        glimpse = internal.rb.get("glimpse", registers[:, i, :])
        glimpse = glimpse.reshape((glimpse.shape[0],) + (internal.image_width, internal.image_width))

        salience_input = internal.rb.get("salience_input", registers[:, i, :])
        salience_input = salience_input.reshape(
            (salience_input.shape[0],) + internal.salience_input_shape)

        salience = internal.rb.get("salience", registers[:, i, :])
        salience = salience.reshape(
            (salience.shape[0],) + internal.salience_output_shape)

        digit = internal.rb.get("digit", registers[:, i, :])
        op = internal.rb.get("op", registers[:, i, :])
        acc = internal.rb.get("acc", registers[:, i, :])

        actions = rollouts.a[:, i, :]

        print("Start of rollout {}.".format(i))
        for t in range(rollouts.T):
            print("t={}".format(t))
            action_idx = int(np.argmax(actions[t, :]))
            print("action={}".format(internal.action_names[action_idx]))
            print("digit: ", digit[t])
            print("op: ", op[t])
            print("acc: ", acc[t])
            print(image_to_string(glimpse[t]))
            print("\n")
            print(image_to_string(salience_input[t]))
            print("\n")
            print(image_to_string(salience[t]))
            print("\n")


def build_env():
    train = GridArithmeticDataset(n_examples=cfg.n_train)
    val = GridArithmeticDataset(n_examples=cfg.n_val)
    test = GridArithmeticDataset(n_examples=cfg.n_val)

    external = RegressionEnv(train, val, test)

    if cfg.ablation == 'bad_wiring':
        internal = GridArithmeticBadWiring()
    elif cfg.ablation == 'no_classifiers':
        internal = GridArithmeticNoClassifiers()
    elif cfg.ablation == 'no_ops':
        internal = GridArithmeticNoOps()
    elif cfg.ablation == 'no_modules':
        internal = GridArithmeticNoModules()
    elif cfg.ablation == 'easy':
        internal = GridArithmeticEasy()
    else:
        internal = GridArithmetic()

    env = CompositeEnv(external, internal)
    env.obs_is_image = True
    return env


def build_policy(env, **kwargs):
    if cfg.ablation == 'bad_wiring':
        action_selection = ProductDist(Softmax(11), Normal(), Normal(), Normal())
    elif cfg.ablation == 'no_classifiers':
        action_selection = ProductDist(Softmax(9), Softmax(10, one_hot=0), Softmax(10, one_hot=0), Softmax(10, one_hot=0))
    elif cfg.ablation == 'no_ops':
        action_selection = ProductDist(Softmax(11), Normal(), Normal(), Normal())
    elif cfg.ablation == 'no_modules':
        action_selection = ProductDist(Softmax(11), Normal(), Normal(), Normal())
    else:
        action_selection = Softmax(env.actions_dim)
        return DiscretePolicy(action_selection, env.obs_shape, **kwargs)
    return Policy(action_selection, env.obs_shape, **kwargs)


config = Config(
    build_env=build_env,

    reductions="A:sum,M:prod,X:max,N:min",
    arithmetic_actions="+,*,max,min,+1",

    curriculum=[dict()],
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.04,
    T=30,
    min_digits=2,
    max_digits=3,
    shape=(2, 2),
    final_reward=True,

    n_train=10000,
    n_val=100,
    use_gpu=False,

    show_op=True,
    reward_window=0.4999,
    salience_action=True,
    salience_input_width=3*14,
    salience_output_width=14,
    initial_salience=False,
    visible_glimpse=False,
    downsample_factor=2,

    ablation='easy',

    build_digit_classifier=lambda: LeNet(128, scope="digit_classifier"),
    build_op_classifier=lambda: LeNet(128, scope="op_classifier"),

    mnist_config=MNIST_CONFIG.copy(
        eval_step=100,
        max_steps=100000,
        patience=5000,
        threshold=0.001,
        include_blank=True,
        use_gpu=True,
        gpu_allow_growth=True,
    ),

    salience_config=MNIST_SALIENCE_CONFIG.copy(
        eval_step=100,
        max_steps=100000,
        patience=5000,
        threshold=0.001,
        render_hook=salience_render_hook(),
        use_gpu=True,
        gpu_allow_growth=True,
    ),

    log_name='grid_arithmetic',
    render_rollouts=grid_arithmetic_render_rollouts,
)


class GridArithmeticDataset(RegressionDataset):
    reductions = Param()
    shape = Param()
    min_digits = Param()
    max_digits = Param()
    base = Param()
    op_loc = Param()
    loss_type = Param("2-norm")
    largest_digit = Param(9)
    downsample_factor = Param(2)
    show_op = Param(True)

    reductions_dict = {
        "sum": sum,
        "prod": np.product,
        "max": max,
        "min": min,
        "len": len,
    }

    def __init__(self, **kwargs):
        self.image_width = int(28 / self.downsample_factor)
        assert 1 <= self.base <= 10
        assert self.min_digits <= self.max_digits
        assert np.product(self.shape) >= self.max_digits + 1

        self.s = s = int(28 / self.downsample_factor)

        if ":" not in self.reductions:
            self.reductions = {'A': self.reductions_dict[self.reductions.strip()]}
            self.show_op = False
        else:
            _reductions = {}
            delim = ',' if ',' in self.reductions else ' '
            for pair in self.reductions.split(delim):
                char, key = pair.split(':')
                _reductions[char] = self.reductions_dict[key]
            self.reductions = _reductions

        op_symbols = sorted(self.reductions)
        emnist_x, emnist_y, symbol_map = load_emnist(
            cfg.data_dir, op_symbols, balance=True,
            downsample_factor=self.downsample_factor)
        emnist_x = emnist_x.reshape(-1, s, s)
        emnist_y = np.squeeze(emnist_y, 1)

        reductions = {symbol_map[k]: v for k, v in self.reductions.items()}

        symbol_reps = DataContainer(emnist_x, emnist_y)

        mnist_x, mnist_y, _ = load_emnist(
            cfg.data_dir, list(range(self.base)), balance=True,
            downsample_factor=self.downsample_factor)
        mnist_x = mnist_x.reshape(-1, s, s)
        mnist_y = np.squeeze(mnist_y, 1)

        digit_reps = DataContainer(mnist_x, mnist_y)
        blank_element = np.zeros((s, s))

        x, y = self.make_dataset(
            self.shape, self.min_digits, self.max_digits, self.base,
            blank_element, digit_reps, symbol_reps,
            reductions, self.n_examples, self.op_loc, self.show_op,
            one_hot_output=self.loss_type == "xent", largest_digit=self.largest_digit)

        super(GridArithmeticDataset, self).__init__(x, y)

    @staticmethod
    def make_dataset(
            shape, min_digits, max_digits, base, blank_element,
            digit_reps, symbol_reps, functions, n_examples, op_loc, show_op,
            one_hot_output, largest_digit):

        new_X, new_Y = [], []

        size = np.product(shape)

        m, n = blank_element.shape
        if op_loc is not None:
            _op_loc = np.ravel_multi_index(op_loc, shape)

        for j in range(n_examples):
            nd = np.random.randint(min_digits, max_digits+1)

            indices = np.random.choice(size, nd+1, replace=False)

            if op_loc is not None and show_op:
                indices[indices == _op_loc] = indices[0]
                indices[0] = _op_loc

            env = np.tile(blank_element, shape)
            locs = zip(*np.unravel_index(indices, shape))
            locs = [(slice(i*m, (i+1)*m), slice(j*n, (j+1)*n)) for i, j in locs]
            op_loc, *digit_locs = locs

            symbol_x, symbol_y = symbol_reps.get_random()
            func = functions[int(symbol_y)]

            if show_op:
                env[op_loc] = symbol_x

            ys = []

            for loc in digit_locs:
                x, y = digit_reps.get_random()
                ys.append(y)
                env[loc] = x

            new_X.append(env)
            y = func(ys)

            if one_hot_output:
                _y = np.zeros(largest_digit+2)
                if y > largest_digit:
                    _y[-1] = 1.0
                else:
                    _y[int(y)] = 1.0
                y = _y

            new_Y.append(y)

        new_X = np.array(new_X).astype('f')

        if one_hot_output:
            new_Y = np.array(new_Y).astype('f')
        else:
            new_Y = np.array(new_Y).astype('i').reshape(-1, 1)

        return new_X, new_Y


def classifier_head(x, base):
    x = tf.stop_gradient(x)
    x = tf.argmax(x, 1)
    x = tf.expand_dims(x, 1)
    x = tf.where(tf.equal(x, base), -1*tf.ones_like(x), x)
    x = tf.cast(x, tf.float32)
    return x


class GridArithmetic(InternalEnv):
    _action_names = ['>', '<', 'v', '^', 'classify_digit', 'classify_op']

    @property
    def input_shape(self):
        return tuple(s*self.image_width for s in self.shape)

    arithmetic_actions = Param()
    shape = Param()
    base = Param()
    start_loc = Param()
    downsample_factor = Param()
    visible_glimpse = Param()
    salience_action = Param()
    salience_input_width = Param()
    salience_output_width = Param()
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
        self.image_width = int(28 / self.downsample_factor)

        _arithmetic_actions = {}
        delim = ',' if ',' in self.arithmetic_actions else ' '
        for key in self.arithmetic_actions.split(delim):
            _arithmetic_actions[key] = self.arithmetic_actions_dict[key]
        self.arithmetic_actions = _arithmetic_actions

        if self.salience_action:
            self.action_names = (
                self._action_names +
                ['update_salience'] +
                sorted(self.arithmetic_actions.keys())
            )
        else:
            self.action_names = self._action_names + sorted(self.arithmetic_actions.keys())

        self.salience_input_shape = (self.salience_input_width,) * 2
        self.salience_output_shape = (self.salience_output_width,) * 2

        self.actions_dim = len(self.action_names)
        self._init_networks()
        self._init_rb()

        super(GridArithmetic, self).__init__()

    def _init_rb(self):
        values = (
            [0., 0., -1., 0., 0., -1.] +
            [np.zeros(self.salience_output_width**2, dtype='f')] +
            [np.zeros(self.image_width**2, dtype='f')] +
            [np.zeros(self.salience_input_width**2, dtype='f')]
        )

        min_values = [0, 10, 0, 0, 0, 0] + [0.] * (self.salience_output_width**2)
        max_values = (
            [9, 12, 999, self.shape[1], self.shape[0], self.actions_dim] +
            [1.] * (self.salience_output_width**2)
        )

        if self.visible_glimpse:
            min_values.extend([0] * self.image_width**2)
            max_values.extend([1] * self.image_width**2)

            self.rb = RegisterBank(
                'GridArithmeticRB',
                'digit op acc fovea_x fovea_y prev_action salience glimpse', 'salience_input', values=values,
                output_names='acc', no_display='glimpse salience salience_input',
                min_values=min_values, max_values=max_values
            )
        else:
            self.rb = RegisterBank(
                'GridArithmeticRB',
                'digit op acc fovea_x fovea_y prev_action salience', 'glimpse salience_input', values=values,
                output_names='acc', no_display='glimpse salience salience_input',
                min_values=min_values, max_values=max_values
            )

    def _init_networks(self):
        digit_config = cfg.mnist_config.copy(
            classes=list(range(self.base)),
            downsample_factor=self.downsample_factor,
            build_function=cfg.build_digit_classifier,
            stopping_function=lambda val_record: -val_record['reward']
        )

        self.digit_classifier = cfg.build_digit_classifier()
        self.digit_classifier.set_pretraining_params(
            digit_config,
            name_params='classes downsample_factor n_train threshold',
            directory=cfg.model_dir + '/mnist_pretrained/'
        )

        op_config = cfg.mnist_config.copy(
            classes=list(self.op_classes),
            downsample_factor=self.downsample_factor,
            build_function=cfg.build_op_classifier,
            n_train=60000,
            stopping_function=lambda val_record: -val_record['reward']
        )

        self.op_classifier = cfg.build_op_classifier()
        self.op_classifier.set_pretraining_params(
            op_config,
            name_params='classes downsample_factor n_train threshold',
            directory=cfg.model_dir + '/mnist_pretrained/',
        )

        self.classifier_head = classifier_head

        if self.salience_action:
            def build_salience_detector(output_width=self.salience_output_width):
                return SalienceMap(
                    cfg.max_digits, MLP([cfg.n_units, cfg.n_units, cfg.n_units], scope="salience_detector"),
                    (output_width, output_width),
                    std=cfg.std, flatten_output=True
                )

            salience_config = cfg.salience_config.copy(
                std=0.1,
                output_width=self.salience_output_width,
                image_width=self.salience_input_width,
                downsample_factor=cfg.downsample_factor,
                min_digits=1,
                max_digits=cfg.max_digits+1,
                build_function=build_salience_detector,
                n_units=101
            )

            with salience_config:
                self.salience_detector = build_salience_detector()

            self.salience_detector.set_pretraining_params(
                salience_config,
                name_params='min_digits max_digits image_width n_units '
                            'downsample_factor output_width',
                directory=cfg.model_dir + '/mnist_salience_pretrained'
            )
        else:
            self.salience_detector = None

    def _build_update_glimpse(self, fovea_y, fovea_x):
        top_left = tf.concat([fovea_y, fovea_x], axis=-1) * self.image_width
        inp = self.input_ph[..., None]
        glimpse = extract_glimpse_numpy_like(
            inp, (self.image_width, self.image_width), top_left)
        glimpse = tf.reshape(glimpse, (-1, self.image_width**2), name="glimpse")
        return glimpse

    def _build_update_salience(self, update_salience, salience, salience_input, fovea_y, fovea_x):
        top_left = tf.concat([fovea_y, fovea_x], axis=-1) * self.image_width
        glimpse = extract_glimpse_numpy_like(
            self.padded_input, (self.salience_input_width, self.salience_input_width), top_left)

        new_salience = self.salience_detector(glimpse, self.salience_output_shape, False)
        new_salience = tf.reshape(new_salience, (-1, self.salience_output_width**2))

        new_salience_input = tf.reshape(glimpse, (-1, self.salience_input_width**2))

        salience = (1 - update_salience) * salience + update_salience * new_salience
        salience_input = (1 - update_salience) * salience_input + update_salience * new_salience_input
        return salience, salience_input

    def _build_update_storage(self, glimpse, prev_digit, classify_digit, prev_op, classify_op):
        digit = self.classifier_head(self.digit_classifier(glimpse, self.base + 1, False), self.base)
        new_digit = (1 - classify_digit) * prev_digit + classify_digit * digit

        op = self.classifier_head(self.op_classifier(glimpse, len(self.op_classes) + 1, False), len(self.op_classes))
        new_op = (1 - classify_op) * prev_op + classify_op * op

        return new_digit, new_op

    def _build_update_fovea(self, right, left, down, up, fovea_y, fovea_x):
        fovea_x = (1 - right - left) * fovea_x + \
            right * (fovea_x + 1) + \
            left * (fovea_x - 1)
        fovea_y = (1 - down - up) * fovea_y + \
            down * (fovea_y + 1) + \
            up * (fovea_y - 1)
        fovea_y = tf.clip_by_value(fovea_y, 0, self.shape[0]-1)
        fovea_x = tf.clip_by_value(fovea_x, 0, self.shape[1]-1)
        return fovea_y, fovea_x

    def _build_return(
            self, digit, op, acc, fovea_x, fovea_y,
            prev_action, salience, glimpse, salience_input):

        with tf.name_scope("GridArithmetic"):
            new_registers = self.rb.wrap(
                digit=tf.identity(digit, "digit"),
                op=tf.identity(op, "op"),
                acc=tf.identity(acc, "acc"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                prev_action=tf.identity(prev_action, "prev_action"),
                salience=tf.identity(salience, "salience"),
                salience_input=tf.identity(salience_input, "salience_input"),
                glimpse=glimpse)

        rewards = self.build_rewards(new_registers)

        return (
            tf.fill((tf.shape(digit)[0], 1), 0.0),
            rewards,
            new_registers)

    def build_init(self, r):
        self.build_placeholders(r)

        self.pad_offset = (int(self.salience_input_width/2 - self.image_width/2),) * 2
        target_height = int(self.input_ph.shape[1]) + 2 * self.pad_offset[0]
        target_width = int(self.input_ph.shape[2]) + 2 * self.pad_offset[1]
        inp = self.input_ph[..., None]
        self.padded_input = resize_image_with_crop_or_pad(inp, target_height, target_width)

        _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        batch_size = tf.shape(self.input_ph)[0]

        # init fovea
        if self.start_loc is not None:
            fovea_y = tf.fill((batch_size, 1), self.start_loc[0])
            fovea_x = tf.fill((batch_size, 1), self.start_loc[1])
        else:
            fovea_y = tf.random_uniform(
                tf.shape(fovea_y), 0, self.shape[0], dtype=tf.int32)
            fovea_x = tf.random_uniform(
                tf.shape(fovea_x), 0, self.shape[1], dtype=tf.int32)

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

        _, _, new_r = self._build_return(
            digit, op, acc, fovea_x, fovea_y, _prev_action, salience, glimpse, salience_input)
        return new_r

    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        (right, left, down, up, classify_digit, classify_op,
            update_salience, *arithmetic_actions) = self.unpack_actions(a)

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

        action = tf.cast(tf.reshape(tf.argmax(a, axis=1), (-1, 1)), tf.float32)

        return self._build_return(digit, op, acc, fovea_x, fovea_y, action, salience, glimpse, salience_input)


class GridArithmeticEasy(GridArithmetic):
    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _prev_action, _salience, _glimpse, _salience_input = self.rb.as_tuple(r)

        (right, left, down, up, classify_digit, classify_op,
            update_salience, *arithmetic_actions) = self.unpack_actions(a)

        salience = _salience
        salience_input = _salience_input
        if self.salience_action:
            salience, salience_input = self._build_update_salience(
                update_salience, _salience, _salience_input, _fovea_y, _fovea_x)

        op = self.classifier_head(self.op_classifier(_glimpse, len(self.op_classes) + 1, False), len(self.op_classes))
        op = (1 - classify_op) * _op + classify_op * op

        new_digit_factor = classify_digit
        for action in arithmetic_actions:
            new_digit_factor += action

        digit = self.classifier_head(self.digit_classifier(_glimpse, self.base + 1, False), self.base)
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

        action = tf.cast(tf.reshape(tf.argmax(a, axis=1), (-1, 1)), tf.float32)

        return self._build_return(digit, op, acc, fovea_x, fovea_y, action, salience, glimpse, salience_input)


class GridArithmeticBadWiring(GridArithmetic):
    action_names = [
        '>', '<', 'v', '^', 'classify_digit', 'classify_op',
        '+', '+1', '*', '=', '+ arg', '* arg', '= arg']

    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)

        (right, left, down, up, classify_digit, classify_op,
         add, inc, multiply, store, add_arg, mult_arg, store_arg) = self.unpack_actions(a)

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (add_arg + _acc) + \
            multiply * (mult_arg * _acc) + \
            inc * (_acc + 1) + \
            store * store_arg

        glimpse = self.build_update_glimpse(_fovea_y, _fovea_x)

        digit, op = self.build_update_storage(
            glimpse, _digit, classify_digit, _op, classify_op)

        fovea_y, fovea_x = self.build_update_fovea(
            right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(digit, op, acc, fovea_x, fovea_y, glimpse)


class GridArithmeticNoClassifiers(GridArithmetic):
    action_names = ['>', '<', 'v', '^', '+', '+1', '*', '=', '+ arg', '* arg', '= arg']

    def init_networks(self):
        return

    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)

        (right, left, down, up, add, inc, multiply, store,
         add_arg, mult_arg, store_arg) = self.unpack_actions(a)

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (add_arg + _acc) + \
            multiply * (mult_arg * _acc) + \
            inc * (_acc + 1) + \
            store * store_arg

        glimpse = self.build_update_glimpse(_fovea_y, _fovea_x)

        fovea_y, fovea_x = self.build_update_fovea(
            right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(_digit, _op, acc, fovea_x, fovea_y, glimpse)


class GridArithmeticNoOps(GridArithmetic):
    action_names = ['>', '<', 'v', '^', 'classify_digit', 'classify_op', '=', '= arg']

    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)

        (right, left, down, up, classify_digit, classify_op,
         store, store_arg) = self.unpack_actions(a)

        acc = (1 - store) * _acc + store * store_arg

        glimpse = self.build_update_glimpse(_fovea_y, _fovea_x)

        digit, op = self.build_update_storage(
            glimpse, _digit, classify_digit, _op, classify_op)

        fovea_y, fovea_x = self.build_update_fovea(
            right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(digit, op, acc, fovea_x, fovea_y, glimpse)


class GridArithmeticNoModules(GridArithmetic):
    action_names = ['>', '<', 'v', '^', '=', '= arg']

    def init_networks(self):
        return

    def build_step(self, t, r, a):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)
        right, left, down, up, store, store_arg = self.unpack_actions(a)

        acc = (1 - store) * _acc + store * store_arg

        glimpse = self.build_update_glimpse(_fovea_y, _fovea_x)

        fovea_y, fovea_x = self.build_update_fovea(
            right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(_digit, _op, acc, fovea_x, fovea_y, glimpse)
