set -ex

ARGS="--n-repeats=5 --ppn=8 --max-hosts=1 --cpp=4 --kind=slurm --pmem=3700 --gpu-set=0,1 --ignore-gpu=False --step-time-limit=3hours --n-raw-features=0"
SHORT_ARGS="--n-repeats=1 --ppn=8 --max-hosts=1 --cpp=4 --kind=slurm --pmem=3700 --gpu-set=0,1 --ignore-gpu=False --step-time-limit=3hours --n-raw-features=0 --error-on-timeout=False"
PRETRAIN_ARGS="--mode=pretrained"
NAME_PREFIX="cnn_pretrained_pure_op"
LONG_TIME="--wall-time=6hours --cleanup-time=30mins --slack-time=30mins"
SHORT_TIME="--wall-time=20mins --cleanup-time=1min --slack-time=1min"

python cnn_op.py --task=A --name="$NAME_PREFIX"_short_B $SHORT_ARGS $PRETRAIN_ARGS $SHORT_TIME

python cnn_op.py --task=A --name="$NAME_PREFIX"_A $ARGS $PRETRAIN_ARGS $LONG_TIME
python cnn_op.py --task=B --name="$NAME_PREFIX"_B $ARGS $PRETRAIN_ARGS $LONG_TIME
python cnn_op.py --task=C --name="$NAME_PREFIX"_C $ARGS $PRETRAIN_ARGS $LONG_TIME
python cnn_op.py --task=D --name="$NAME_PREFIX"_D $ARGS $PRETRAIN_ARGS $LONG_TIME

PRETRAIN_ARGS="--mode=standard"
NAME_PREFIX="cnn_op"

python cnn_op.py --task=A --name="$NAME_PREFIX"_short_B $SHORT_ARGS $PRETRAIN_ARGS $SHORT_TIME

python cnn_op.py --task=A --name="$NAME_PREFIX"_A $ARGS $PRETRAIN_ARGS $LONG_TIME
python cnn_op.py --task=B --name="$NAME_PREFIX"_B $ARGS $PRETRAIN_ARGS $LONG_TIME
python cnn_op.py --task=C --name="$NAME_PREFIX"_C $ARGS $PRETRAIN_ARGS $LONG_TIME
python cnn_op.py --task=D --name="$NAME_PREFIX"_D $ARGS $PRETRAIN_ARGS $LONG_TIME