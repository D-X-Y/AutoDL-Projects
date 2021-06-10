#!/bin/bash
# bash ./scripts/experimental/train-vit.sh cifar10 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for dataset and random-seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
rseed=$2

save_dir=./outputs/${dataset}/vit-experimental

python --version

python ./exps/basic/xmain.py --save_dir ${save_dir} --rand_seed ${rseed} \
	--train_data_config ./configs/data.yaml/${dataset}.train \
	--valid_data_config ./configs/data.yaml/${dataset}.test \
	--data_path $TORCH_HOME/cifar.python
