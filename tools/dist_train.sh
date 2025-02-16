#!/usr/bin/env bash
set -e
set -x

CONFIG=$1
GPUS=${GPUS:-4}
echo $GPUS 
python -m torch.distributed.launch --nproc_per_node=$GPUS \
  $(dirname "$0")/train.py --config-file $CONFIG ${@:2}
