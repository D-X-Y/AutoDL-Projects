#!/bin/bash
#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01 #
#####################################################
# [mars6] CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/X-X/train-shapes.sh 00000-05000 12 777
# [mars6] bash ./scripts-search/X-X/train-shapes.sh 05001-10000 12 777
# [mars20] bash ./scripts-search/X-X/train-shapes.sh 10001-14500 12 777
# [mars20] bash ./scripts-search/X-X/train-shapes.sh 14501-19500 12 777
# bash ./scripts-search/X-X/train-shapes.sh 19501-23500 12 777
# bash ./scripts-search/X-X/train-shapes.sh 23501-27500 12 777
# bash ./scripts-search/X-X/train-shapes.sh 27501-30000 12 777
# bash ./scripts-search/X-X/train-shapes.sh 30001-32767 12 777
#
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for start-and-end, hyper-parameters-opt-file, and seeds"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

srange=$1
opt=$2
all_seeds=$3
cpus=4

save_dir=./output/NAS-BENCH-202/

OMP_NUM_THREADS=${cpus} python exps/NAS-Bench-201/xshapes.py \
	--mode new --srange ${srange} --hyper ${opt} --save_dir ${save_dir} \
	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
	--splits   1       0       0        0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python/ImageNet16 \
	--workers ${cpus} \
	--seeds ${all_seeds}