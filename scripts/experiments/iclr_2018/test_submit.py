import subprocess
import os
import argparse

import dps
from dps.utils import cd

parser = argparse.ArgumentParser(
    "Test reinforcement learning on grid_arithmetic. "
    "Run for each new commit to make sure that it still works."
)
parser.add_argument("kind", choices="parallel slurm".split())
parser.add_argument("length", choices="short long".split())

args = parser.parse_args()

if args.kind == "parallel":
    pass
elif args.kind == "slurm":
    with cd(os.path.dirname(dps.__file__)):
        sha = subprocess.check_output("git rev-parse --verify --short HEAD".split()).decode().strip()

    if args.length == "long":
        cmd = (
            'python rl_main.py --name=test_combined_long_commit_{} --wall-time=24hours --seed=23123798 --n-repeats=5 '
            '--ppn=8 --max-hosts=4 --cpp=4 --kind=slurm --cleanup-time=30mins --slack-time=30mins '
            '--gpu-set=0,1 --ignore-gpu=True --pmem=3700'.format(sha)
        )
    elif args.length == "short":
        cmd = (
            'python rl_main.py --name=test_combined_short_commit_{} --wall-time=20mins --seed=23123798 --n-repeats=5 '
            '--ppn=8 --max-hosts=4 --cpp=4 --kind=slurm --cleanup-time=2mins --slack-time=2mins '
            '--gpu-set=0,1 --ignore-gpu=True --pmem=3700'.format(sha)
        )

    print(cmd)

    subprocess.run(cmd.split())
else:
    raise Exception("NotImplemented.")
