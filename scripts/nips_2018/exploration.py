import clify
import numpy as np
from dps.env.advanced import yolo_rl
from dps.datasets import EMNIST_ObjectDetection


def prepare_func():
    from dps import cfg
    from dps.train import PolynomialScheduleHook

    fragment = [
        dict(obj_exploration=0.2,),
        dict(obj_exploration=0.1,),
        dict(obj_exploration=0.05,),
    ]

    area_weight_factor = cfg.area_weight

    cfg.hooks = [
        PolynomialScheduleHook(
            attr_name="area_weight",
            query_name="best_COST_reconstruction",
            base_configs=fragment, tolerance=None,
            initial_value=area_weight_factor,
            scale=area_weight_factor, power=1.0)
    ]


distributions = dict(
    order=["box obj attr", "obj attr box"],
    area_weight=list(np.e ** np.linspace(-8, -3, 10)),
    nonzero_weight=list(np.linspace(5, 50, 10))
)

config = yolo_rl.good_config.copy(
    prepare_func=prepare_func,
    patience=2500,
    lr_schedule=1e-4,
    render_step=100000,
    max_overlap=40,
    hooks=[],
    n_val=16,
    eval_step=1000,

    dynamic_partition=True,
    fix_values=dict(),

    curriculum=[
        dict(
            fix_values=dict(obj=1), dynamic_partition=False,
            patience=100000, max_steps=10000),
    ],
)

# Create the datasets if necessary.
with config:
    train = EMNIST_ObjectDetection(n_examples=int(config.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(config.n_val), shuffle=True, example_range=(0.9, 1.))

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=distributions, n_param_settings=32)
