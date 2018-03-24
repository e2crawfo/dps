from dps import cfg
from dps.env.advanced import yolo_rl
from dps.datasets.atari import AtariAutoencodeDataset
from dps.train import PolynomialScheduleHook, GeometricScheduleHook


class Env(object):
    def __init__(self):
        train = AtariAutoencodeDataset(n_examples=int(cfg.n_train))
        val = AtariAutoencodeDataset(n_examples=int(cfg.n_val))
        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


def build_env():
    return Env()


config = yolo_rl.experimental_config.copy(
    log_name="yolo_atari",
    game="SpaceInvadersNoFrameskip-v4",
    policy=None,
    n_train=5000,
    samples_per_frame=5,
    image_shape=(60, 60),
    build_env=build_env,
    n_val=16,
    density=0.1,
    show_images=16,
    max_hw=3.0,
    min_hw=0.25,
    hooks=[
        PolynomialScheduleHook(
            "nonzero_weight", "best_COST_reconstruction",
            base_configs=[
                dict(obj_exploration=0.2,),
                dict(obj_exploration=0.1,),
                dict(obj_exploration=0.05,),
            ],
            tolerance=0.5, scale=5, power=1., initial_value=1.0),
    ],
    nonzero_weight=0.0,
    area_weight=0.02,
)

continuation_config = config.copy(
    curriculum=[
        dict(fix_values=dict(obj=1), do_train=False, load_path="/home/eric/Dropbox/experiment_data/active/yolo_rl/space_invaders_area/weights/best_of_stage_0"),
    ],
    hooks=[
        PolynomialScheduleHook(
            "area_weight", "best_COST_reconstruction",
            base_configs=[
                dict(obj_exploration=0.2,),
                dict(obj_exploration=0.1,),
                dict(obj_exploration=0.05,),
            ],
            tolerance=0.5, scale=0.05, power=1., initial_value=1),
            # tolerance=0.5, scale=0.05, power=1., initial_value=0.05),
    ],
    nonzero_weight=50.0,
)

large_config = config.copy(
    load_path="/home/eric/Dropbox/experiment_data/active/yolo_rl/space_invaders_first/weights/best_of_stage_12",
    image_shape=(210, 160),
    do_train=False,
    samples_per_frame=0,
    n_train=1,
    n_val=16,
    render_hook=yolo_rl.YoloRL_RenderHook(N=16),
    hooks=[],
    curriculum=[dict()],
    use_gpu=False,
    dynamic_partition=True,
    density=0.01
)
