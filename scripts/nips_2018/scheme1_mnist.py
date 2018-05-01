"""
In training scheme 1, we don't anneal anything. There is an initial round of training without RL,
followed by a fragment wherein the exploration is gradually decreased.

There are 3 main hyper-parameters:

    * area_weight during stage 0
    * area_weight during remaining stages
    * nonzero_weight during remaining stages

"""
import clify
import numpy as np
from dps.env.advanced import yolo_rl
from dps.datasets import EmnistObjectDetectionDataset


def prepare_func():
    from dps import cfg
    cfg.curriculum[0]["area_weight"] = cfg.stage0_area_weight


distributions = dict(
    nonzero_weight=[70., 75., 80., 85., 90],
    area_weight=list(np.linspace(0.1, 0.8, 5)),
    stage0_area_weight=[.01, .02, .03, .04],
)


config = yolo_rl.good_config.copy(
    prepare_func=prepare_func,
    patience=10000,
    render_step=100000,
    lr_schedule=1e-4,
    max_overlap=40,
    hooks=[],
    n_val=16,
    eval_step=1000,
    max_steps=100000,

    fixed_values=dict(),

    curriculum=[
        dict(fixed_values=dict(obj=1), max_steps=10000),
        dict(obj_exploration=0.2,),
        dict(obj_exploration=0.1,),
        dict(obj_exploration=0.1, lr_schedule=1e-5),
        dict(obj_exploration=0.1, lr_schedule=1e-6),
    ],
)

# Create the datasets if necessary.
with config:
    train = EmnistObjectDetectionDataset(n_examples=int(config.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EmnistObjectDetectionDataset(n_examples=int(config.n_val), shuffle=True, example_range=(0.9, 1.))

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=distributions)
