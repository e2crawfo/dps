import clify
from dps.env.advanced import yolo_rl


def prepare_func():
    from dps import cfg
    from dps.train import PolynomialScheduleHook

    fragment = [
        dict(obj_exploration=0.2,),
        dict(obj_exploration=0.1,),
        dict(obj_exploration=0.05,),
    ]

    kwargs = dict(
        query_name="best_COST_reconstruction",
        base_configs=fragment, tolerance=None,
        initial_value=10.0, scale=10.0, power=1.0)

    schedules = cfg.get("schedule", "").split()

    hooks = []

    if "nonzero" in schedules:
        hooks.append(
            PolynomialScheduleHook(attr_name="nonzero_weight", **kwargs))
    elif "area" in schedules:
        hooks.append(
            PolynomialScheduleHook(attr_name="area_weight", **kwargs))

    cfg.hooks = hooks


_grid = [
    dict(),
    dict(order="obj attr cls box"),
    # dict(n_train=1e4),
]


grid = []
grid += [dict(schedule="area", **g) for g in _grid]
grid += [dict(schedule="nonzero", **g) for g in _grid]
grid += [dict(schedule="nonzero area", **g) for g in _grid]


config = yolo_rl.good_experimental_config.copy(
    prepare_func=prepare_func,
    patience=2500,
    lr_schedule=1e-4,

    dynamic_partition=True,
    fix_values=dict(),
    nonzero_weight=10.0,
    area_weight=10.0,

    curriculum=[
        dict(
            fix_values=dict(obj=1), dynamic_partition=False, patience=100000,
            area_weight=0.0, nonzero_weight=0.0, max_steps=2500),
    ],

    sub_image_size_std=0.4,
    max_hw=3.0,
    min_hw=0.25,
    max_overlap=40,
    image_shape=(50, 50),
    hooks=[],
)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=grid)
