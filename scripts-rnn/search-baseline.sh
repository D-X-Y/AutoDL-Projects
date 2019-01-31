#!/usr/bin/env sh
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for the GPU"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

gpus=$1
epoch=50
SAVED=./snapshots/NAS-RNN/Search-Baseline-${epoch}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/rnn/train_rnn_search.py \
	--data_path ./data/data/penn \
	--save_path ${SAVED} \
	--epochs ${epoch} \
	--config_path ./configs/NAS-PTB-BASE.config \
	--print_freq 200
