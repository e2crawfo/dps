import argparse
import subprocess
import os


parser = argparse.ArgumentParser("Extract only the high-level data from a set of experiments, leaving behind details from individual experiments.")
parser.add_argument("source")
parser.add_argument("dest")

args = parser.parse_args()
source = args.source
dest = args.dest

for d in os.listdir(source):
    directory = os.path.join(source, d)
    while directory.endswith('/'):
        directory = directory[:-1]

    if os.path.isdir(directory) and 'experiments' in os.listdir(directory):
        cmd = 'rsync -av {} {} --exclude experiments'.format(directory, dest)
        subprocess.run(cmd, shell=True)