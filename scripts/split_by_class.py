"""
Download data set from:

https://www.nist.gov/itl/iad/image-group/emnist-dataset

wget http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip
unzip matlab.zip
cd emnist

run this script from there.


"""

from pathlib import Path
import dill
import numpy as np
from scipy.io import loadmat
import shutil


chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "


def char_map(value):
    """ Assumes value between 0 and 1. """
    if value >= 1:
        value = 1 - 1e-6
    n_bins = len(chars)
    bin_id = int(value * n_bins)
    return chars[bin_id]


def image_to_string(array):
    if array.ndim == 1:
        array = array.reshape(-1, int(np.sqrt(array.shape[0])))
    image = [char_map(value) for value in array.flatten()]
    image = np.reshape(image, array.shape)
    return '\n'.join(''.join(c for c in row) for row in image)


def split_by_class(filename):
    dir_name = Path(Path(filename).stem)
    try:
        shutil.rmtree(str(dir_name))
    except FileExistsError:
        pass

    dir_name.mkdir(exist_ok=False, parents=False)

    emnist = loadmat(filename)
    train, test, _ = emnist['dataset'][0, 0]
    train_x, train_y, _ = train[0, 0]
    test_x, test_y, _ = test[0, 0]

    y = np.concatenate((train_y, test_y), 0)
    x = np.concatenate((train_x, test_x), 0)
    # Give them the right orientation so that plt.imshow(x[0]) just works.
    x = np.moveaxis(x.reshape(-1, 28, 28), 1, 2).reshape(-1, 28**2)
    x = x.astype('f') / 255.0

    for i in sorted(set(y.flatten())):
        keep = train_y == i
        x_i = x[keep.flatten(), :]
        print(i)
        if i >= 36:
            char = chr(i-36+ord('a'))
        elif i >= 10:
            char = chr(i-10+ord('A'))
        else:
            char = str(i)
        print(char)
        print(image_to_string(x_i[0, :]))

        path_i = dir_name / (str(i) + '.pkl')
        with path_i.open('wb') as f:
            dill.dump(x_i, f, protocol=dill.HIGHEST_PROTOCOL)

    return x, y


if __name__ == "__main__":
    split_by_class('./emnist-byclass.mat')
