import clify
from dps.env.advanced import yolo_rl


def prepare_func():
    from dps import cfg
    from dps.train import PolynomialScheduleHook, GeometricScheduleHook
    fragment = [
        dict(obj_exploration=0.2,),
        dict(obj_exploration=0.1,),
        dict(obj_exploration=0.05,),
    ]

    kwargs = dict(
        attr_name="nonzero_weight", query_name="best_COST_reconstruction",
        base_configs=fragment, tolerance=0.1, initial_value=1.0)

    schedule = None

    if cfg.schedule == "exp2":
        print("Geometric schedule")
        schedule = GeometricScheduleHook(**kwargs)
    elif cfg.schedule == "exp3":
        print("Geometric schedule")
        schedule = GeometricScheduleHook(multiplier=3., **kwargs)
    elif cfg.schedule == "exp4":
        print("Geometric schedule")
        schedule = GeometricScheduleHook(multiplier=4., **kwargs)
    elif cfg.schedule == "poly2":
        print("Polynomial2 schedule")
        schedule = PolynomialScheduleHook(scale=1., power=2., **kwargs)
    elif cfg.schedule == "poly3":
        print("Polynomial3 schedule")
        schedule = PolynomialScheduleHook(scale=1., power=3., **kwargs)
    elif cfg.schedule == "poly4":
        print("Polynomial4 schedule")
        schedule = PolynomialScheduleHook(scale=1., power=4., **kwargs)

    if schedule:
        cfg.hooks = [schedule]


grid = [
    dict(schedule="exp3"),
    dict(schedule="exp4"),
    dict(max_hw=2.0),
    dict(max_hw=1.5),
    dict(order="obj cls attr box"),
    dict(n_train=1e4),
]

grid = grid + [dict(patience=2500, **d) for d in grid]

config = yolo_rl.good_experimental_config.copy(
    prepare_func=prepare_func,
    schedule="exp2",
)

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

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=grid)
