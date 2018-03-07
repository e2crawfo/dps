from dps.env.advanced import yolo_rl, yolo_rl_simple
from dps.utils.tf import ScopedFunction


class PassthroughDecoder(ScopedFunction):
    def _call(self, inp, output_shape, is_training):
        _, input_glimpses = inp
        return input_glimpses


config = yolo_rl.config.copy(
    log_name="yolo_passthrough",
    get_updater=yolo_rl_simple.get_updater,
    build_object_decoder=PassthroughDecoder,
    C=1,
    A=2,
    characters=[0],
    # characters=[0, 1, 2],
    object_shape=(28, 28),
    anchor_boxes=[[28, 28]],
    sub_image_shape=(28, 28),

    use_input_attention=True,
    decoders_output_logits=False,

    fixed_backbone=False,
    fixed_next_step=False,
    fixed_object_decoder=False,
)
