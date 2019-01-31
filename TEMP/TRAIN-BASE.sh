#!/usr/bin/env sh
# bash scripts-nas/TRAIN-BASE.sh 0 DMS_V1 cifar10 nocut init-channel layers
if [ "$#" -ne 6 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 6 parameters for the GPUs, the architecture, the dataset, the config, the initial channel, and the number of layers"
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
dataset=$3
config=$4
C=$5
N=$6
SAVED=./snapshots/NAS/${arch}-${C}-${N}-${dataset}-${config}-E600

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/train_base.py \
	--data_path $TORCH_HOME/cifar.python \
	--dataset ${dataset} --arch ${arch} \
	--save_path ${SAVED} \
	--grad_clip 5 \
	--init_channels ${C} --layers ${N} \
	--model_config ./configs/nas-cifar-cos-${config}.config \
	--print_freq 100 --workers 8
