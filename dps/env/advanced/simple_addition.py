import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.env import CompositeEnv, InternalEnv
from dps.env.supervised import IntegerRegressionEnv
from dps.datasets import Dataset, ArrayFeature
from dps.vision.attention import apply_gaussian_filter
from dps.utils import Param, Config


def build_env():
    train = SimpleAdditionDataset(n_examples=cfg.n_train)
    val = SimpleAdditionDataset(n_examples=cfg.n_val)

    external = IntegerRegressionEnv(train, val)
    internal = SimpleAddition()

    return CompositeEnv(external, internal)


config = Config(
    build_env=build_env,
    T=30,
    curriculum=[
        dict(width=1),
        dict(width=2),
        dict(width=3),
    ],
    base=10,
    final_reward=True,
    n_controller_units=32,
    log_name='simple_addition',
)


class SimpleAdditionDataset(Dataset):
    width = Param()
    base = Param()

    @property
    def obs_shape(self):
        return (2 * self.width + 1,)

    @property
    def action_shape(self):
        return (1,)

    @property
    def features(self):
        if self._features is not None:
            return self._features

        self._features = [
            ArrayFeature("x", self.obs_shape, dtype=np.uint8),
            ArrayFeature("y", self.action_shape, dtype=np.uint8),
        ]
        return self._features

    def _make(self):
        x = np.random.randint(0, self.base, size=(self.n_examples, 2*self.width+1))
        y = x[:, :1] + x[:, -1:]

        for _x, _y in zip(x, y):
            self._write_example(x=_x, y=_y)


class SimpleAddition(InternalEnv):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm1 = vision', 'wm2 = vision',
                    'output = vision', 'output = wm1 + wm2', 'no-op/stop']
    width = Param()

    def __init__(self, **kwargs):

        self.rb = RegisterBank(
            'SimpleAdditionRB',
            'fovea vision wm1 wm2 output', None,
            values=[0.] * 5, output_names='output')

        super(SimpleAddition, self).__init__(**kwargs)

    @property
    def input_shape(self):
        return (2*self.width+1,)

    def build_init(self, r):
        self.maybe_build_placeholders()

        fovea, vision, wm1, wm2, output = self.rb.as_tuple(r)
        std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(
            np.linspace(-self.width, self.width, 2*self.width+1, dtype='f'),
            dtype=tf.float32)
        vision = apply_gaussian_filter(fovea, std, locations, self.input)

        new_registers = self.rb.wrap(
            fovea=fovea, vision=vision, wm1=wm1, wm2=wm2, output=output)

        return new_registers

    def build_step(self, t, r, a):
        _fovea, _vision, _wm1, _wm2, _output = self.rb.as_tuple(r)

        actions = self.unpack_actions(a)
        inc_fovea, dec_fovea, vision_to_wm1, vision_to_wm2, vision_to_output, add, no_op = actions

        fovea = (1 - inc_fovea - dec_fovea) * _fovea + inc_fovea * (_fovea + 1) + dec_fovea * (_fovea - 1)
        fovea = tf.clip_by_value(fovea, -self.width, self.width)
        wm1 = (1 - vision_to_wm1) * _wm1 + vision_to_wm1 * _vision
        wm2 = (1 - vision_to_wm2) * _wm2 + vision_to_wm2 * _vision
        output = (1 - vision_to_output - add) * _output + vision_to_output * _vision + add * (_wm1 + _wm2)

        std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f'), dtype=tf.float32)
        vision = apply_gaussian_filter(fovea, std, locations, self.input)

        new_registers = self.rb.wrap(
            fovea=fovea, vision=vision, wm1=wm1, wm2=wm2, output=output)

        reward = self.build_reward(new_registers, actions)
        done = tf.zeros(tf.shape(r)[:-1])[..., None]

        return done, reward, new_registers
