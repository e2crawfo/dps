set -ex

ARGS="--n-repeats=5 --max-hosts=2 --ppn=16 --cpp=2 --kind=slurm --pmem=7800"
SHORT_ARGS="--n-repeats=1 --max-hosts=2 --ppn=16 --cpp=2 --kind=slurm --pmem=7800"
LONG_TIME="--wall-time=24hours --cleanup-time=30mins --slack-time=30mins"
SHORT_TIME="--wall-time=20mins --cleanup-time=1min --slack-time=1min"

EXTRA_ARGS="--ablation=alternate --ignore-gpu=True --gpu-set=0,1"
# EXTRA_ARGS="--ablation=alternate"
NAME_PREFIX="rl_alternate"

python rl_main.py --name="$NAME_PREFIX"_combined_short $SHORT_ARGS $SHORT_TIME $EXTRA_ARGS

python rl_main.py --name="$NAME_PREFIX"_combined $ARGS $LONG_TIME $EXTRA_ARGS
python rl_main.py --name="$NAME_PREFIX"_sum --reductions=sum $ARGS $LONG_TIME $EXTRA_ARGS
python rl_main.py --name="$NAME_PREFIX"_prod --reductions=prod $ARGS $LONG_TIME $EXTRA_ARGS
python rl_main.py --name="$NAME_PREFIX"_max --reductions=max $ARGS $LONG_TIME $EXTRA_ARGS
python rl_main.py --name="$NAME_PREFIX"_min --reductions=min $ARGS $LONG_TIME $EXTRA_ARGS