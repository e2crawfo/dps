from dps.env.advanced.yolo import config as yolo_config
from dps.env.advanced.yolo import FullyConvolutional

""" Testing YOLO on MNIST dataset when there is only 1 output grid-cell.

Verdict:

    Works great! Easily gets to 1.0 mAP, and bounding boxes look fantastic.
    Even works with poorly chosen anchor boxes.

"""


class TinyYoloBackbone1D(FullyConvolutional):
    def __init__(self):
        layout = [
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
            dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
        ]
        super(TinyYoloBackbone1D, self).__init__(layout, check_output_shape=True)


def build_fcn():
    return TinyYoloBackbone1D()


config = yolo_config.copy(
    log_name="yolo_single_output",
    build_fully_conv_net=build_fcn,
    characters=[0, 1],
    min_chars=1,
    max_chars=1,
    H=1,
    W=1,
    anchor_boxes=[[23, 14]],
)
