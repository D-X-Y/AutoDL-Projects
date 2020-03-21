#!/bin/bash
# bash ./scripts-search/NAS-Bench-201/test-weights.sh cifar10-valid 1
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for dataset and use_12_epoch"
  exit 1
fi

if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python exps/NAS-Bench-201/test-weights.py \
	--base_path $HOME/.torch/NAS-Bench-201-v1_1-096897 \
	--dataset $1 \
	--use_12 $2
