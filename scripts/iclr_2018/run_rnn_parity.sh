set -ex

ARGS="--n-repeats=5 --ppn=8 --max-hosts=1 --cpp=4 --kind=slurm --pmem=3700 --gpu-set=0,1 --ignore-gpu=False --step-time-limit=3hours --n-raw-features=0"
SHORT_ARGS="--n-repeats=1 --ppn=8 --max-hosts=1 --cpp=4 --kind=slurm --pmem=3700 --gpu-set=0,1 --ignore-gpu=False --step-time-limit=3hours --n-raw-features=0 --error-on-timeout=False"
PRETRAIN_ARGS="--pretrain=True"
NAME_PREFIX="rnn_pretrained_pure_parity"
LONG_TIME="--wall-time=6hours --cleanup-time=30mins --slack-time=30mins"
SHORT_TIME="--wall-time=20mins --cleanup-time=1min --slack-time=1min"

python rnn_parity.py --task=B --name="$NAME_PREFIX"_short_B $SHORT_ARGS $PRETRAIN_ARGS $SHORT_TIME

python rnn_parity.py --task=B --name="$NAME_PREFIX"_B $ARGS $PRETRAIN_ARGS $LONG_TIME
python rnn_parity.py --task=C --name="$NAME_PREFIX"_C $ARGS $PRETRAIN_ARGS $LONG_TIME

PRETRAIN_ARGS="--pretrain=False"
NAME_PREFIX="rnn_pure_parity"

python rnn_parity.py --task=B --name="$NAME_PREFIX"_short_B $SHORT_ARGS $PRETRAIN_ARGS $SHORT_TIME

python rnn_parity.py --task=B --name="$NAME_PREFIX"_B $ARGS $PRETRAIN_ARGS $LONG_TIME
python rnn_parity.py --task=C --name="$NAME_PREFIX"_C $ARGS $PRETRAIN_ARGS $LONG_TIME