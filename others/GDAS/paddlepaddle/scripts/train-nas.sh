#!/bin/bash
# bash ./scripts/base-train.sh 0 cifar-10 ResNet110
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for GPU and dataset and the-model-name"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

GPU=$1
dataset=$2
model=$3

save_dir=snapshots/${dataset}-${model}

export FLAGS_fraction_of_gpu_memory_to_use="0.005"
export FLAGS_free_idle_memory=True

CUDA_VISIBLE_DEVICES=${GPU} python train_cifar.py \
	--data_path $TORCH_HOME/cifar.python/${dataset}-python.tar.gz \
	--log_dir ${save_dir} \
	--dataset ${dataset}  \
	--model_name ${model} \
	--lr 0.025 --epochs 600 --batch_size 96 --step_each_epoch 521
