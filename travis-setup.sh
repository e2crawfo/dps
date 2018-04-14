#!/usr/bash
set -ev

pip install -r requirements_dev.txt
pip install tensorflow==1.6.0
pip install -e .

pip install pipdeptree
pipdeptree

mkdir data

echo "\nDownloading and processing emnist data..."
time python scripts/download.py emnist data -q
rm data/matlab.zip

echo "\nDownloading and processing omniglot data..."
time python scripts/download.py omniglot data -q

echo "\nDownloading and installing gnu-parallel..."
wget https://mirror.csclub.uwaterloo.ca/gnu/parallel/parallel-20170622.tar.bz2
tar -xjvf parallel-20170622.tar.bz2
cd parallel-20170622
./configure --prefix="$TRAVIS_BUILD_DIR" && make && make install
cd ..
rm -rf parallel-20170622.tar.bz2 parallel-20170622
