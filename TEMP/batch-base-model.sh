#!/usr/bin/env sh
set -e
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for the GPUs"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

gpus=$1

bash ./scripts-nas/train-model.sh ${gpus} AmoebaNet 0

bash ./scripts-nas/train-model.sh ${gpus} NASNet    0

bash ./scripts-nas/train-model.sh ${gpus} DARTS_V1  0

bash ./scripts-nas/train-model.sh ${gpus} DARTS_V2  0
