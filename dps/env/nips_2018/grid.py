from dps import cfg
from dps.datasets import GridEMNIST_ObjectDetection
from dps.utils import Config


class Nips2018Grid(object):
    def __init__(self):
        train = GridEMNIST_ObjectDetection(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = GridEMNIST_ObjectDetection(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


config = Config(
    log_name="nips_2018_grid",
    build_env=Nips2018Grid,

    # dataset params
    use_dataset_cache=True,
    min_chars=25,
    max_chars=25,
    n_sub_image_examples=0,
    draw_shape_grid=(5, 5),
    image_shape=(5*14, 5*14),
    sub_image_shape=(14, 14),
    draw_offset="random",
    spacing=(0, 0),
    characters=list(range(10)),
    colours="white",

    xent_loss=True,

    n_train=1e5,
    n_val=2**7,
    n_test=2**7,
)
