#!/usr/bin/env sh
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for the GPU and the architecture"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

gpus=$1
arch=$2
SAVED=./snapshots/NAS-RNN/Search-${arch}-WT2

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/rnn/train_rnn_base.py \
	--arch ${arch} \
	--save_path ${SAVED} \
	--config_path ./configs/NAS-WT2-BASE.config \
	--print_freq 300
