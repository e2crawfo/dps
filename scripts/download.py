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


# This link seems not to work anymore...
# emnist_url = 'https://cloudstor.aarnet.edu.au/plus/index.php/s/54h3OuGJhFLwAlQ/download'

emnist_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'


template = 'emnist-byclass-{}-{}-idx{}-ubyte.gz'

emnist_gz_names = [
    template.format('test', 'images', 3),
    template.format('test', 'labels', 1),
    template.format('train', 'images', 3),
    template.format('train', 'labels', 1)
]


def download_emnist(data_dir):
    """
    Download the emnist data. Result is that a directory called "emnist_raw"
    is created inside `data_dir` which contains 4 files.

    Parameters
    ----------
    path: str
        Path to directory where files should be stored.

    """
    emnist_raw_dir = os.path.join(data_dir, "emnist_raw")
    os.makedirs(emnist_raw_dir, exist_ok=True)

    with cd(emnist_raw_dir):
        if not os.path.exists('gzip.zip'):
            print("Downloading...")
            command = "wget --output-document=gzip.zip {}".format(emnist_url).split()
            subprocess.run(command, check=True)
        else:
            print("Found existing copy of gzip.zip, not downloading.")

        print("Extracting...")
        for fname in emnist_gz_names:
            if not os.path.exists(fname):
                subprocess.run('unzip gzip.zip gzip/{}'.format(fname), shell=True, check=True)
                shutil.move('gzip/{}'.format(fname), '.')
            else:
                print("{} already exists, skipping extraction.".format(fname))

        try:
            shutil.rmtree('gzip')
        except FileNotFoundError:
            pass

    return emnist_raw_dir


def _emnist_load_helper(path_img, path_lbl):
    with gzip.open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        labels = array("B", file.read())

    with gzip.open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))

        image_data = array("B", file.read())

    images = np.zeros((size, rows * cols))

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    return np.array(images), np.array(labels)


def process_emnist(data_dir, quiet):
    """
    Download emnist data if it hasn't already been downloaded. Do some
    post-processing to put it in a more useful format. End result is a directory
    called `emnist-byclass` which contains a separate pklz file for each emnist
    class.

    Pixel values of stored images are uint8 values up to 255.
    Images for each class are put into a numpy array with shape (n_images_in_class, 28, 28).
    This numpy array is pickled and stored in a zip file with name <class char>.pklz.

    Parameters
    ----------
    data_dir: str
         Directory where files should be stored.

    """
    with cd(data_dir):
        if _validate_emnist('emnist'):
            print("EMNIST data seems to be present already.")
            return
        else:
            try:
                shutil.rmtree('emnist')
            except FileNotFoundError:
                pass

    raw_dir = download_emnist(data_dir)

    with cd(raw_dir):
        images, labels = _emnist_load_helper(emnist_gz_names[0], emnist_gz_names[1])
        images1, labels1 = _emnist_load_helper(emnist_gz_names[2], emnist_gz_names[3])

    with cd(data_dir):
        os.makedirs('emnist', exist_ok=False)

        print("Processing...")
        with cd('emnist'):
            x = np.concatenate((images, images1), 0)
            y = np.concatenate((labels, labels1), 0)

            # Give images the right orientation so that plt.imshow(x[0]) just works.
            x = np.moveaxis(x.reshape(-1, 28, 28), 1, 2)

            for i in sorted(set(y.flatten())):
                keep = y == i
                x_i = x[keep.flatten(), :]
                if i >= 36:
                    char = chr(i-36+ord('a'))
                elif i >= 10:
                    char = chr(i-10+ord('A'))
                else:
                    char = str(i)

                if quiet >= 2:
                    pass
                elif quiet == 1:
                    print(char)
                elif quiet <= 0:
                    print(char)
                    print(image_to_string(x_i[0, ...]))

                file_i = char + '.pklz'
                with gzip.open(file_i, 'wb') as f:
                    dill.dump(x_i, f, protocol=dill.HIGHEST_PROTOCOL)

    print("Done setting up EMNIST data.")
    return x, y


omniglot_alphabets = [
    'Bengali',
    'Balinese',
    'Arcadian',
    'Tibetan',
    'Braille',
    'Burmese_(Myanmar)',
    'Malay_(Jawi_-_Arabic)',
    'Cyrillic',
    'Hebrew',
    'Atemayar_Qelisayer',
    'Latin',
    'Mkhedruli_(Georgian)',
    'Mongolian',
    'Asomtavruli_(Georgian)',
    'Oriya',
    'Kannada',
    'Futurama',
    'Inuktitut_(Canadian_Aboriginal_Syllabics)',
    'Tifinagh',
    'Old_Church_Slavonic_(Cyrillic)',
    'Grantha',
    'Greek',
    'N_Ko',
    'Blackfoot_(Canadian_Aboriginal_Syllabics)',
    'Malayalam',
    'Manipuri',
    'Gujarati',
    'Anglo-Saxon_Futhorc',
    'Korean',
    'Sanskrit',
    'Sylheti',
    'Aurek-Besh',
    'Ojibwe_(Canadian_Aboriginal_Syllabics)',
    'Gurmukhi',
    'Avesta',
    'Angelic',
    'Japanese_(katakana)',
    'Japanese_(hiragana)',
    'ULOG',
    'Early_Aramaic',
    'Tagalog',
    'Glagolitic',
    'Syriac_(Serto)',
    'Alphabet_of_the_Magi',
    'Armenian',
    'Syriac_(Estrangelo)',
    'Keble',
    'Atlantean',
    'Ge_ez',
    'Tengwar'
]


def _validate_omniglot(path):
    if not os.path.isdir(path):
        return False

    with cd(path):
        return set(os.listdir(path)) == set(omniglot_alphabets)


def process_omniglot(data_dir, quiet):
    try:
        omniglot_dir = process_path(os.path.join(data_dir, 'omniglot'))

        if _validate_omniglot(omniglot_dir):
            print("Omniglot data seems to be present already.")
            return
        else:
            try:
                shutil.rmtree(omniglot_dir)
            except FileNotFoundError:
                pass

        os.makedirs(omniglot_dir, exist_ok=False)

        with cd(omniglot_dir):
            subprocess.run("git clone https://github.com/brendenlake/omniglot --depth=1".split(), check=True)

            with cd('omniglot/python'):
                zip_ref = zipfile.ZipFile('images_evaluation.zip', 'r')
                zip_ref.extractall('.')
                zip_ref.close()

                zip_ref = zipfile.ZipFile('images_background.zip', 'r')
                zip_ref.extractall('.')
                zip_ref.close()

            subprocess.run('mv omniglot/python/images_background/* .', shell=True, check=True)
            subprocess.run('mv omniglot/python/images_evaluation/* .', shell=True, check=True)
        print("Done setting up Omniglot data.")
    finally:
        try:
            shutil.rmtree(os.path.join(omniglot_dir, 'omniglot'))
        except FileNotFoundError:
            pass


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
        process_emnist(args.path, args.q)
        if args.shape:
            shape = tuple(int(i) for i in args.shape.split(','))
            convert_emnist_and_store(args.path, shape)
    elif args.kind == 'omniglot':
        process_omniglot(args.path, args.q)
    elif args.kind == 'backgrounds':
        download_backgrounds(args.path)
    else:
        raise Exception("NotImplemented")
