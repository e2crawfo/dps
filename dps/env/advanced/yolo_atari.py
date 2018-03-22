from dps import cfg
from dps.env.advanced import yolo_rl
from dps.datasets.atari import AtariAutoencodeDataset


class Env(object):
    def __init__(self):
        train = AtariAutoencodeDataset(n_examples=int(cfg.n_train))
        val = AtariAutoencodeDataset(n_examples=int(cfg.n_val))
        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


def build_env():
    return Env()


config = yolo_rl.good_experimental_config.copy(
    log_name="yolo_atari",
    game="SpaceInvadersNoFrameskip-v4",
    policy=None,
    n_train=5000,
    samples_per_frame=5,
    image_shape=(60, 60),
    build_env=build_env,
    n_val=16,
    density=0.01,
    show_images=16,
    max_hw=3.0,
    min_hw=0.25,
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
