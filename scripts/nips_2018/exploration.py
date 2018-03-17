import clify

from dps.train import PolynomialScheduleHook, GeometricScheduleHook
from dps.env.advanced import yolo_rl


static_decoder_config = dict(
    build_object_decoder=yolo_rl.StaticObjectDecoder,
    C=1,
    A=2,
    characters=[0],
    decoder_logit_scale=100.,
)

static_decoder_multiclass_config = dict(
    C=2,
    characters=[0, 1],
    **static_decoder_config,
)

poly_schedule3_config = dict(
    hooks=[
        PolynomialScheduleHook(
            "nonzero_weight", "best_COST_reconstruction",
            base_configs=[
                dict(obj_exploration=0.2,),
                dict(obj_exploration=0.1,),
                dict(obj_exploration=0.05,),
            ],
            tolerance=0.1, scale=1., power=3., initial_value=1.0
        ),
    ],
)

poly_schedule4_config = dict(
    hooks=[
        PolynomialScheduleHook(
            "nonzero_weight", "best_COST_reconstruction",
            base_configs=[
                dict(obj_exploration=0.2,),
                dict(obj_exploration=0.1,),
                dict(obj_exploration=0.05,),
            ],
            tolerance=0.1, scale=1., power=4., initial_value=1.0
        ),
    ],
)

exponential_schedule_config = dict(
    hooks=[
        GeometricScheduleHook(
            "nonzero_weight", "best_COST_reconstruction",
            base_configs=[
                dict(obj_exploration=0.2,),
                dict(obj_exploration=0.1,),
                dict(obj_exploration=0.05,),
            ],
            tolerance=0.1, initial_value=1.0
        ),
    ],
)

grid = [
    poly_schedule3_config,
    poly_schedule4_config,
    exponential_schedule_config,
    dict(max_hw=2.0),
    dict(order="obj cls attr box"),
    dict(n_train=1e4),
]

grid = grid + [dict(patience=2500, **d) for d in grid]

config = yolo_rl.good_experimental_config.copy()

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--task", choices="B C".split(), default='')
# args, _ = parser.parse_known_args()
# if args.task == "B":
#     config.curriculum = [dict(parity='even', n_train=2**17), dict(parity='odd')]
# elif args.task == "C":
#     config.curriculum = [dict(parity='odd')]
# else:
#     raise Exception()

from dps.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
