set -ex

ARGS="--n-repeats=5 --ppn=8 --max-hosts=1 --cpp=4 --kind=slurm --pmem=3700 --gpu-set=0,1 --ignore-gpu=False --step-time-limit=3hours --n-raw-features=0"
SHORT_ARGS="--n-repeats=1 --ppn=8 --max-hosts=1 --cpp=4 --kind=slurm --pmem=3700 --gpu-set=0,1 --ignore-gpu=False --step-time-limit=3hours --n-raw-features=0 --error-on-timeout=False"
LONG_TIME="--wall-time=12hours --cleanup-time=30mins --slack-time=30mins"
# LONG_TIME="--wall-time=6hours --cleanup-time=30mins --slack-time=30mins"
SHORT_TIME="--wall-time=20mins --cleanup-time=1min --slack-time=1min"

PRETRAIN_ARGS="--pretrain=True"
NAME_PREFIX="rnn_pretrained_pure_size"

# python rnn_size.py --task=C --name="$NAME_PREFIX"_short_C $SHORT_ARGS $PRETRAIN_ARGS $SHORT_TIME

# python rnn_size.py --task=A --name="$NAME_PREFIX"_A $ARGS $PRETRAIN_ARGS $LONG_TIME
python rnn_size.py --task=B --name="$NAME_PREFIX"_B $ARGS $PRETRAIN_ARGS $LONG_TIME
python rnn_size.py --task=C --name="$NAME_PREFIX"_C $ARGS $PRETRAIN_ARGS $LONG_TIME
# python rnn_size.py --task=D --name="$NAME_PREFIX"_D $ARGS $PRETRAIN_ARGS $LONG_TIME
# python rnn_size.py --task=E --name="$NAME_PREFIX"_E $ARGS $PRETRAIN_ARGS $LONG_TIME
# python rnn_size.py --task=F --name="$NAME_PREFIX"_F $ARGS $PRETRAIN_ARGS $LONG_TIME

PRETRAIN_ARGS="--pretrain=False"
NAME_PREFIX="rnn_pure_size"

# python rnn_size.py --task=C --name="$NAME_PREFIX"_short_C $SHORT_ARGS $PRETRAIN_ARGS $SHORT_TIME

# python rnn_size.py --task=A --name="$NAME_PREFIX"_A $ARGS $PRETRAIN_ARGS $LONG_TIME
python rnn_size.py --task=B --name="$NAME_PREFIX"_B $ARGS $PRETRAIN_ARGS $LONG_TIME
python rnn_size.py --task=C --name="$NAME_PREFIX"_C $ARGS $PRETRAIN_ARGS $LONG_TIME
# python rnn_size.py --task=D --name="$NAME_PREFIX"_D $ARGS $PRETRAIN_ARGS $LONG_TIME
# python rnn_size.py --task=E --name="$NAME_PREFIX"_E $ARGS $PRETRAIN_ARGS $LONG_TIME
# python rnn_size.py --task=F --name="$NAME_PREFIX"_F $ARGS $PRETRAIN_ARGS $LONG_TIME