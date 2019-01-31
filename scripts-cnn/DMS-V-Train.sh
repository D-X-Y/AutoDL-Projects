#!/usr/bin/env sh
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for the GPUs and the epochs and the cutout"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

gpus=$1
arch=acc2
cutout=$3
dataset=cifar10
epoch=$2
SAVED=./snapshots/NAS/ACC-V2-Search-${arch}-${dataset}-cut${cutout}-${epoch}-E600

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-nas/acc_search_v2.py \
	--data_path $TORCH_HOME/cifar.python \
	--arch ${arch} --dataset ${dataset} --batch_size 128 \
	--save_path ${SAVED} \
	--learning_rate_max 0.05 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 0.0003 \
	--epochs ${epoch} --cutout ${cutout} --validate --grad_clip 5 \
	--init_channels 16 --layers 8 \
	--tau_max 10 --tau_min 4 \
	--model_config ./configs/nas-cifar-cos.config \
	--print_freq 100 --workers 8
