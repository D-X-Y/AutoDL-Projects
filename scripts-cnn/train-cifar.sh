#!/usr/bin/env sh
# bash scripts-cnn/train-cifar.sh 0 GDAS cifar10 cut
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for the GPUs, the architecture, and the dataset-name, and the cutout"
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
cutout=$4
SAVED=./snapshots/NAS/${arch}-${dataset}-${cutout}-E600
#--data_path $TORCH_HOME/cifar.python \

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-cnn/train_base.py \
	--data_path ./data/data/cifar.python \
	--dataset ${dataset} --arch ${arch} \
	--save_path ${SAVED} \
	--grad_clip 5 \
	--init_channels 36 --layers 20 \
	--model_config ./configs/nas-cifar-cos-${cutout}.config \
	--print_freq 100 --workers 8
