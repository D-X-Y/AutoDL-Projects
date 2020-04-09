#!/bin/bash
#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01 #
#####################################################
# bash ./scripts-search/X-X/train-shapes-01.sh 0 4
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for hyper-parameters-opt-file, and seeds"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

srange=00000-32767
opt=01
all_seeds=777
cpus=4

save_dir=./output/NAS-BENCH-202/

SLURM_PROCID=$1 SLURM_NTASKS=$2 OMP_NUM_THREADS=${cpus} python exps/NAS-Bench-201/xshapes.py \
	--mode new --srange ${srange} --hyper ${opt} --save_dir ${save_dir} \
	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
	--splits   1       0       0        0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python/ImageNet16 \
	--workers ${cpus} \
	--seeds ${all_seeds}
