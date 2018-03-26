from dps import cfg
from dps.env.advanced import yolo_rl
from dps.datasets.atari import AtariAutoencodeDataset
from dps.datasets import ImageDataset


class Env(object):
    def __init__(self):
        dset = AtariAutoencodeDataset(n_examples=int(cfg.n_train))
        train = ImageDataset(tracks=[dset.tracks[0][:-cfg.n_val]])
        val = ImageDataset(tracks=[dset.tracks[0][-cfg.n_val:]])
        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


def build_env():
    return Env()


config = yolo_rl.good_config.copy(
    log_name="yolo_atari",
    build_env=build_env,

    game="SpaceInvadersNoFrameskip-v4",
    policy=None,
    n_train=5000,
    samples_per_frame=5,
    image_shape=(60, 60),
    density=0.1,
    show_images=16,

    n_val=16,
    area_weight=0.25,
    nonzero_weight=60.0,

    curriculum=[
        dict(fix_values=dict(obj=1), dynamic_partition=False, max_steps=10000, area_weight=0.01),
        dict(obj_exploration=0.2),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
    ],
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

keyboard_config = config.copy(
    policy="keyboard",
)

throwaway_config = config.copy(
    load_path="/data/dps_data/logs/yolo_atari/exp_yolo_atari_seed=347405995_2018_03_24_16_02_44/weights/best_of_stage_3",
    do_train=False,
    hooks=[],
    curriculum=[dict()],
    use_gpu=True,
    obj_exploration=0.0,
)
