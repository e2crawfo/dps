import shutil
import numpy as np
from scipy.io import loadmat
import dill
import gzip
import argparse
import os
import subprocess
import zipfile

from mnist_arithmetic.emnist import _validate_emnist
from mnist_arithmetic.utils import image_to_string, cd, process_path


emnist_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'


def download_emnist(path):
    """
    Download the emnist data. Result is that a file called `emnist-byclass.mat` is stored in `path`.

    Parameters
    ----------
    path: str
        Path to directory where files should be stored.

    """
    if not os.path.exists('matlab.zip'):
        print("Downloading...")
        subprocess.run(('wget ' + emnist_url).split(), check=True)
    else:
        print("Found existing copy of matlab.zip, not downloading.")

    matlab_zip_path = process_path('./matlab.zip')

    with cd(path):
        print("Extracting...")
        subprocess.run('unzip {} matlab/emnist-byclass.mat'.format(matlab_zip_path), shell=True, check=True)

        shutil.move('matlab/emnist-byclass.mat', '.')
        shutil.rmtree('matlab')


def process_emnist(data_dir, quiet):
    """
    Download emnist data if it hasn't already been downloaded. Do some
    post-processing to put it in a more useful format. End result is a directory
    called `emnist-byclass` which contains a separate pklz file for each emnist
    class.

    Pixel values of stored images are floating points values between 0 and 1.
    Images for each class are put into a floating point numpy array with shape
    (n_images_in_class, 28, 28). This numpy array is pickled and stored in a zip
    file with name <class char>.pklz.

    Parameters
    ----------
    data_dir: str
         Directory where files should be stored.

    """
    emnist_dir = process_path(os.path.join(data_dir, 'emnist'))

    if _validate_emnist(emnist_dir):
        print("EMNIST data seems to be present already, exiting.")
        return
    else:
        try:
            shutil.rmtree(emnist_dir)
        except FileNotFoundError:
            pass

    os.makedirs(emnist_dir, exist_ok=True)

    download_emnist(emnist_dir)

    print("Processing...")
    with cd(emnist_dir):
        emnist = loadmat('emnist-byclass.mat')

        train, test, _ = emnist['dataset'][0, 0]
        train_x, train_y, _ = train[0, 0]
        test_x, test_y, _ = test[0, 0]

        y = np.concatenate((train_y, test_y), 0)
        x = np.concatenate((train_x, test_x), 0)

        # Give images the right orientation so that plt.imshow(x[0]) just works.
        x = np.moveaxis(x.reshape(-1, 28, 28), 1, 2)
        x = x.astype('f') / 255.0

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

        os.remove('emnist-byclass.mat')

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
            print("Omniglot data seems to be present already, exiting.")
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
    parser.add_argument('kind', type=str, choices=['emnist', 'omniglot'])
    parser.add_argument('path', type=str)
    parser.add_argument('-q', action='count')
    args = parser.parse_args()

    if args.kind == 'emnist':
        process_emnist(args.path, args.q)
    elif args.kind == 'omniglot':
        process_omniglot(args.path, args.q)
    else:
        raise Exception("NotImplemented")
