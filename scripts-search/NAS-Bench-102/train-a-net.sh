#!/bin/bash
# bash ./scripts-search/NAS-Bench-102/train-a-net.sh resnet 16 5
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for network, channel, num-of-cells"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

model=$1
channel=$2
num_cells=$3

save_dir=./output/NAS-BENCH-102-4/

OMP_NUM_THREADS=4 python ./exps/NAS-Bench-102/main.py \
	--mode specific-${model} --save_dir ${save_dir} --max_node 4 \
	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
	--use_less 0 \
	--splits         1       0        0              0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python/ImageNet16 \
	--channel ${channel} --num_cells ${num_cells} \
	--workers 4 \
	--seeds 777 888 999
