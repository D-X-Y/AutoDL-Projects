#!/usr/bin/env sh
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for the GPUs and the network and the dataset"
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
dataset=$3
epoch=50
SAVED=./snapshots/NAS/Search-${arch}-${dataset}-cut${cutout}-${epoch}

if [ "$dataset" == "cifar10" ] ;then
  dataset_root=$TORCH_HOME/cifar.python
  print_freq=100
elif [ "$dataset" == "cifar100" ] ;then
  dataset_root=$TORCH_HOME/cifar.python
  print_freq=100
elif [ "$dataset" == "tiered" ] ;then
  dataset_root=$TORCH_HOME/tiered-imagenet
  print_freq=500
else
  echo 'invalid dataset-name :'${dataset}
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpus} python ./exps-cnn/DARTS-Search.py \
	--data_path ${dataset_root} \
	--arch ${arch} \
	--dataset ${dataset} --batch_size 64 \
	--save_path ${SAVED} \
	--learning_rate_max 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 0.0003 \
	--epochs ${epoch} --cutout ${cutout} --validate --grad_clip 5 \
	--init_channels 16 --layers 8 \
	--model_config ./configs/nas-cifar-cos-cut.config \
	--print_freq ${print_freq} --workers 8
