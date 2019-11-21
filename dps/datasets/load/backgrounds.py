import imageio
import numpy as np
import os
import subprocess

from dps import cfg
from dps.utils import cd, resize_image


def background_names():
    backgrounds_dir = os.path.join(cfg.data_dir, 'backgrounds')
    return sorted(
        f.split('.')[0]
        for f in os.listdir(backgrounds_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    )


def hard_background_names():
    backgrounds_dir = os.path.join(cfg.data_dir, 'backgrounds')
    return sorted(
        f.split('.')[0]
        for f in os.listdir(backgrounds_dir)
        if f.endswith('.jpg')
    )


def load_backgrounds(background_names, shape=None):
    if isinstance(background_names, str):
        background_names = background_names.split()

    backgrounds_dir = os.path.join(cfg.data_dir, 'backgrounds')
    backgrounds = []
    for name in background_names:
        f = os.path.join(backgrounds_dir, '{}.jpg'.format(name))
        try:
            b = imageio.imread(f)
        except FileNotFoundError:
            f = os.path.join(backgrounds_dir, '{}.png'.format(name))
            b = imageio.imread(f)

        if shape is not None and b.shape != shape:
            b = resize_image(b, shape)
            b = np.uint8(b)

        backgrounds.append(b)
    return backgrounds


background_url = "https://github.com/e2crawfo/backgrounds.git"


def download_backgrounds(data_dir):
    """
    Download backgrounds. Result is that a file called `emnist-byclass.mat` is stored in `data_dir`.

    Parameters
    ----------
    path: str
        Path to directory where files should be stored.

    """
    with cd(data_dir):
        if not os.path.exists('backgrounds'):
            command = "git clone {}".format(background_url).split()
            subprocess.run(command, check=True)
