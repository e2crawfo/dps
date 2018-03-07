import tensorflow as tf

from dps import cfg
from dps.env.advanced import yolo_rl
from dps.utils.tf import ScopedFunction
from dps.datasets import AutoencodeDataset, EMNIST_ObjectDetection


class StaticObjectDecoder(ScopedFunction):
    """ An object decoder that outputs a learnable image, ignoring input. """
    def __init__(self, initializer=None, **kwargs):
        super(StaticObjectDecoder, self).__init__(**kwargs)
        if initializer is None:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.initializer = initializer

    def _call(self, inp, output_shape, is_training):
        output = 100 * tf.get_variable(
            "learnable_bias",
            shape=output_shape,
            dtype=tf.float32,
            initializer=self.initializer,
            trainable=True,
        )
        return tf.tile(output[None, ...], (tf.shape(inp)[0],) + tuple(1 for s in output_shape))


class YoloRLSimple_Updater(yolo_rl.YoloRL_Updater):
    def _make_datasets(self):
        _train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train), colours="red").x
        train = AutoencodeDataset(_train, image=True, shuffle=True)

        _val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val), colours="red").x
        val = AutoencodeDataset(_val, image=True, shuffle=False)

        self.datasets = dict(train=train, val=val)


def get_updater(env):
    return YoloRLSimple_Updater()


config = yolo_rl.config.copy(
    log_name="yolo_rl_simple",
    get_updater=get_updater,
    build_object_decoder=StaticObjectDecoder,
    C=1,
    A=2,
    characters=[0],
    # characters=[0, 1, 2],
    object_shape=(28, 28),
    anchor_boxes=[[28, 28]],
    sub_image_shape=(28, 28),
    cls_exploration=0.5,
    obj_exploration=0.5,
    box_std=-1,
    max_grad_norm=None,
    lr_schedule=1e-4,

    fixed_backbone=False,
    fixed_next_step=False,
    fixed_object_decoder=False,

    xent_loss=True
)
