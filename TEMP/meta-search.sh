#!/usr/bin/env sh
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for the GPUs and the network and N-way and K-shot"
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
n_way=$3
k_shot=$4
cutout=16
epoch=60
SAVED=./snapshots/NAS/Meta-Search-${arch}-N${n_way}-K${k_shot}-cut${cutout}-${epoch}

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/meta_search.py \
	--data_path $TORCH_HOME/tiered-imagenet \
	--arch ${arch} --n_way ${n_way} --k_shot ${k_shot} \
	--save_path ${SAVED} \
	--learning_rate_max 0.001 --learning_rate_min 0.0001 --momentum 0.9 --weight_decay 0.0003 \
	--epochs ${epoch} --cutout ${cutout} --validate --grad_clip 5 \
	--init_channels 16 --layers 8 \
	--model_config ./configs/nas-cifar-cos-cut.config \
	--print_freq 200 --workers 16
