import shutil
import numpy as np
import dill
import gzip
import argparse
import os
import subprocess
import zipfile
import struct
from array import array

from dps import cfg
from dps.datasets.load import _validate_emnist
from dps.utils import image_to_string, cd, process_path
from dps.datasets.load import convert_emnist_and_store

from dps.datasets.load import mayb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', type=str, choices=['emnist', 'omniglot', 'backgrounds'])
    parser.add_argument('--path', type=str, default=cfg.data_dir)
    parser.add_argument('-q', action='count', default=0)
    parser.add_argument(
        '--shape', default="", type=str,
        help="Only valid when kind=='emnist'. If provided, assumes that emnist "
             "dataset has already been downloaded and processed. Value should be "
             "comma-separated pair of integers. Creates a copy of the emnist dataset, "
             "resized to have the given shape")
    args = parser.parse_args()

    if args.kind == 'emnist':
        if args.shape:
            shape = tuple(int(i) for i in args.shape.split(','))
        else:
            shape = None

        process_emnist(args.path, args.q, shape=shape)
    elif args.kind == 'omniglot':
        process_omniglot(args.path, args.q)
    elif args.kind == 'backgrounds':
        download_backgrounds(args.path)
    else:
        raise Exception("NotImplemented")
