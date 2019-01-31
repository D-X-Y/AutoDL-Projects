#!/usr/bin/env sh
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for the GPUs and the network"
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
cutout=0
dataset=cifar10
epoch=50
SAVED=./snapshots/NAS/ACC-V2-Search-${arch}-${dataset}-cut${cutout}-${epoch}-E600

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/acc_search_v2.py \
	--data_path $TORCH_HOME/cifar.python \
	--arch ${arch} --dataset ${dataset} --batch_size 128 \
	--save_path ${SAVED} \
	--learning_rate_max 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 0.0003 \
	--epochs ${epoch} --cutout ${cutout} --validate --grad_clip 5 \
	--init_channels 16 --layers 8 \
	--model_config ./configs/nas-cifar-cos.config \
	--print_freq 100 --workers 8
