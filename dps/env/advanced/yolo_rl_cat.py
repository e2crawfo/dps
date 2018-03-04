# Test pure categorization

from dps.utils.tf import FullyConvolutional
from dps.env.advanced import yolo_rl


class _ObjectDecoder(FullyConvolutional):
    def __init__(self, **kwargs):
        layout = [
            dict(filters=128, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=256, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=3, kernel_size=3, strides=2, padding="SAME", transpose=True),  # For 28 x 28 output
        ]
        super(_ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


config = yolo_rl.config.copy()

curriculum = [
    yolo_rl.combined_mode
]


config.update(
    log_name="yolo_rl_cat",
    curriculum=curriculum,
    C=1,
    sub_image_shape=(28, 28),
    object_shape=(28, 28),
    anchor_boxes=[[28, 28]],
    build_object_decoder=_ObjectDecoder,
)
