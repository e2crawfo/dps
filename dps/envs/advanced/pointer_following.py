import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.envs import CompositeEnv, InternalEnv
from dps.envs.supervised import SupervisedDataset, IntegerRegressionEnv
from dps.vision.attention import apply_gaussian_filter
from dps.utils import Param, Config


def build_env():
    train = PointerDataset(n_examples=cfg.n_train)
    val = PointerDataset(n_examples=cfg.n_val)
    test = PointerDataset(n_examples=cfg.n_val)
    external = IntegerRegressionEnv(train, val, test)
    internal = Pointer()
    return CompositeEnv(external, internal)


config = Config(
    build_env=build_env,
    T=30,
    curriculum=[dict(width=2, base=10)],
    log_name='pointer',
)


class PointerDataset(SupervisedDataset):
    width = Param()
    base = Param()

    def __init__(self, **kwargs):
        width = self.width
        x = np.random.randint(0, self.base, size=(self.n_examples, 2*width+1))
        x[:, width] = np.random.randint(-width, width+1, size=self.n_examples)
        y = x[range(x.shape[0]), x[:, width]+width].reshape(-1, 1)
        super(PointerDataset, self).__init__(x, y)


class Pointer(InternalEnv):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm = vision', 'no-op/stop']

    width = Param()

    def __init__(self, **kwargs):
        self.rb = RegisterBank(
            'PointerRB', 'fovea vision wm', None, values=([0.0] * 3), output_names='wm',
        )
        super(Pointer, self).__init__(**kwargs)

    @property
    def input_shape(self):
        return (2*self.width+1,)

    def build_init(self, r):
        self.maybe_build_placeholders()

        fovea, vision, wm = self.rb.as_tuple(r)

        diag_std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f').reshape(-1, 1))
        vision = apply_gaussian_filter(fovea, diag_std, locations, self.input_ph)

        new_registers = self.rb.wrap(fovea=fovea, vision=vision, wm=wm)

        return new_registers

    def build_step(self, t, r, a):
        _fovea, _vision, _wm = self.rb.as_tuple(r)

        inc_fovea, dec_fovea, vision_to_wm, no_op = self.unpack_actions(a)

        fovea = (1 - inc_fovea - dec_fovea) * _fovea + inc_fovea * (_fovea + 1) + dec_fovea * (_fovea - 1)
        fovea = tf.clip_by_value(fovea, -self.width, self.width)
        wm = (1 - vision_to_wm) * _wm + vision_to_wm * _vision

        diag_std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f').reshape(-1, 1))
        vision = apply_gaussian_filter(fovea, diag_std, locations, self.input_ph)

        new_registers = self.rb.wrap(fovea=fovea, vision=vision, wm=wm)

        rewards = self.build_reward(new_registers)

        return (
            tf.fill((tf.shape(r)[0], 1), 0.0),
            rewards,
            new_registers)
