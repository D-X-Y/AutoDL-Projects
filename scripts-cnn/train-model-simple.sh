#!/usr/bin/env sh
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for the GPUs and the architecture"
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
dataset=cifar10
SAVED=./snapshots/NAS/${arch}-${dataset}-E100

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/train_base.py \
	--data_path $TORCH_HOME/cifar.python \
	--dataset ${dataset} --arch ${arch} \
	--save_path ${SAVED} \
	--grad_clip 5 \
	--model_config ./configs/nas-cifar-cos-simple.config \
	--print_freq 100 --workers 8
