#!/bin/bash
# bash ./scripts-search/search-depth-gumbel.sh cifar10 ResNet110 CIFARX 0.57 777
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for the dataset and the-model-name and the-optimizer and FLOP-ratio and the-random-seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
model=$2
optim=$3
expected_FLOP_ratio=$4
rseed=$5

bash ./scripts-search/search-depth-cifar.sh ${dataset} ${model} ${optim} 0.1 5 ${expected_FLOP_ratio} ${rseed}
