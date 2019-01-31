#!/usr/bin/env sh
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for the GPUs, the architecture, and the channel and the layers"
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
dataset=imagenet
channels=$3
layers=$4
SAVED=./snapshots/NAS/${arch}-${dataset}-C${channels}-L${layers}-E250

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/train_base.py \
	--data_path $TORCH_HOME/ILSVRC2012 \
	--dataset ${dataset} --arch ${arch} \
	--save_path ${SAVED} \
	--grad_clip 5 \
	--init_channels ${channels} --layers ${layers} \
	--model_config ./configs/nas-imagenet.config \
	--print_freq 200 --workers 20
