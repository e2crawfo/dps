import subprocess
import os
import argparse
import socket

import dps
from dps.utils import cd

parser = argparse.ArgumentParser(
    "Test reinforcement learning on grid_arithmetic. "
    "Run for each new commit to make sure that it still works."
)
parser.add_argument("kind", choices="parallel slurm".split())
parser.add_argument("length", choices="short long".split())
parser.add_argument("queue", choices="cpu gpu".split())

args = parser.parse_args()

if args.kind == "parallel":
    pass
elif args.kind == "slurm":
    with cd(os.path.dirname(dps.__file__)):
        sha = subprocess.check_output("git rev-parse --verify --short HEAD".split()).decode().strip()

    hostname = socket.gethostname()
    if "gra" in hostname:
        resources = "--max-hosts=4 --ppn=8 --pmem=3800"
        gpu = "--gpu-set=0,1 --ignore-gpu=True"
    elif "cedar" in hostname:
        if args.queue == "gpu":
            resources = "--max-hosts=5 --ppn=6 --pmem=7700"
        else:
            resources = "--max-hosts=4 --ppn=8 --pmem=3800"
        gpu = "--gpu-set=0,1,2,3 --ignore-gpu=True"
    else:
        raise Exception("Unknown host: {}".format(hostname))

    if args.queue == "cpu":
        gpu = ""

    name = "test_combined_{}_{}_{}_commit_{}".format(args.kind, args.length, args.queue, sha)

    if args.length == "long":
        time = '--wall-time=24hours --cleanup-time=30mins --slack-time=30mins'
    else:
        time = '--wall-time=20mins --cleanup-time=2mins --slack-time=2mins '

    cmd = (
        'python rl_main.py --name={name} '
        '--seed=23123798 --n-repeats=5 {time} {resources} {gpu} --cpp=4 '
        '--kind=slurm'.format(name=name, time=time, resources=resources, gpu=gpu)
    )

    print(cmd)

    subprocess.run(cmd.split())
else:
    raise Exception("NotImplemented.")
