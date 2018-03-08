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

    fix_values=dict(cls=1),

    lr_schedule=1e-6,
)

multi_config = config.copy(
    H=3,
    W=3,
    obj_nonzero_weight=40.0,
    obj_exploration=0.30,
    cls_exploration=0.30,

    object_shape=(14, 14),
    anchor_boxes=[[14, 14]],
    sub_image_shape=(14, 14),

    box_std=-1.,
    attr_std=0.0,
    minimize_kl=True,

    cell_yx_target_mean=0.5,
    cell_yx_target_std=1.0,
    hw_target_mean=1.0,
    hw_target_std=1.0,
    attr_target_mean=0.0,
    attr_target_std=1.0,
)


experimental_config = multi_config.copy(
    H=7,
    W=7,
    obj_exploration=0.05,
    obj_nonzero_weight=20.0,
    prediction_map_weight=0.1,
)
