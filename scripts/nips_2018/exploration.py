import clify
import numpy as np
from dps.env.advanced import yolo_rl
from dps.datasets import EMNIST_ObjectDetection
from dps.train import PolynomialScheduleHook


def prepare_func():
    from dps import cfg
    stage0 = cfg.curriculum[0]
    if cfg.initial_stage == "normal":
        stage0.update(fix_values=dict(obj=1))
    elif cfg.initial_stage == "explore":
        stage0.update(obj_exploration=1.0, obj_default=0.5)
    else:
        raise Exception()
    stage0["area_weight"] = cfg.stage0_area_weight


distributions = dict(
    area_weight=list(np.linspace(0.1, 0.4, 3)),
    stage0_area_weight=[.01, .025],
    initial_stage=["normal", "explore"],
    box_std=[0.0, 0.1],
)

fragment = [
    dict(obj_exploration=0.2,),
    dict(obj_exploration=0.1,),
    dict(obj_exploration=0.05,),
]

config = yolo_rl.small_test_config.copy(
    prepare_func=prepare_func,
    patience=2500,
    lr_schedule=1e-4,
    render_step=100000,
    max_overlap=40,
    n_val=16,
    eval_step=1000,

    fix_values=dict(),

    curriculum=[
        dict(fixed_obj=True, nonzero_weight=0, patience=10000, max_steps=10000)
    ],

    hooks=[
        PolynomialScheduleHook(
            attr_name="nonzero_weight",
            query_name="best_COST_reconstruction",
            base_configs=fragment, tolerance=2,
            initial_value=90, scale=5, power=1.0)
    ]
)

# Create the datasets if necessary.
with config:
    train = EMNIST_ObjectDetection(n_examples=int(config.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(config.n_val), shuffle=True, example_range=(0.9, 1.))

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=distributions)
