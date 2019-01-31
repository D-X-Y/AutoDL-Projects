#!/usr/bin/env sh
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for the GPU and the epochs and tau-max and tau-min"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

gpus=$1
epoch=$2
tau_max=$3
tau_min=$4
SAVED=./snapshots/NAS-RNN/Search-Accelerate-tau_${tau_max}_${tau_min}-${epoch}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/rnn/acc_rnn_search.py \
	--data_path ./data/data/penn \
	--save_path ${SAVED} \
	--epochs ${epoch} \
	--tau_max ${tau_max} --tau_min ${tau_min} \
	--config_path ./configs/NAS-PTB-BASE.config \
	--print_freq 200
