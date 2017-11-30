#!/usr/bash
set -ev
pip install -r requirements_dev.txt
MNIST_ARITHMETIC_PATH=$(python -c "import mnist_arithmetic; from pathlib import Path; print(Path(mnist_arithmetic.__file__).parent.parent)")
DATA_DIR="$TRAVIS_BUILD_DIR"/data
mkdir "$DATA_DIR"

echo "\nDownloading and processing emnist data..."
time python "$MNIST_ARITHMETIC_PATH"/download.py emnist "$DATA_DIR"
rm matlab.zip

echo "\nDownloading and processing omniglot data..."
time python "$MNIST_ARITHMETIC_PATH"/download.py omniglot "$DATA_DIR"

echo "\nDownloading and installing gnu-parallel..."
OLD_WD=$PWD
cd "$TRAVIS_BUILD_DIR"
echo "$PWD"
wget http://mirror.sergal.org/gnu/parallel/parallel-20170622.tar.bz2
tar -xjvf parallel-20170622.tar.bz2
cd parallel-20170622
./configure --prefix="$TRAVIS_BUILD_DIR" && make && make install
cd ..
rm -rf parallel-20170622.tar.bz2 parallel-20170622
cd "$OLD_WD"
