#!/bin/bash
#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01 #
#####################################################
# SLURM_PROCID=0 SLURM_NTASKS=6 bash ./scripts-search/X-X/train-shapes-v2.sh 12 777
#
# SLURM_PROCID=0 SLURM_NTASKS=2 bash ./scripts-search/X-X/train-shapes.sh 31000-32767 90 777
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

#srange=01000-03999,04050-05000,06000-09000,11000-14500,15000-18500,20000-23500,25000-27500,29000-30000
#srange=00000-00999,04000-04049,05001-05999,09001-10999,14501-14999,18501-19999,23501-24999,27501-28999,30001-32767
srange=00000-00999,04000-04049,05001-05999,09001-10999,14501-14999,18501-19999,23501-24999,27501-28999,30001-30999
opt=$1
all_seeds=$2
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
