#!/bin/bash
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01                          #
##############################################################################
# [saturn1] CUDA_VISIBLE_DEVICES=0 bash scripts/NATS-Bench/train-topology.sh 00000-02000 200 "777 888 999"
# [saturn1] CUDA_VISIBLE_DEVICES=0 bash scripts/NATS-Bench/train-topology.sh 02000-04000 200 "777 888 999"
# [saturn1] CUDA_VISIBLE_DEVICES=1 bash scripts/NATS-Bench/train-topology.sh 04000-06000 200 "777 888 999"
# [saturn1] CUDA_VISIBLE_DEVICES=1 bash scripts/NATS-Bench/train-topology.sh 06000-08000 200 "777 888 999"
#
# CUDA_VISIBLE_DEVICES=0 bash scripts/NATS-Bench/train-topology.sh 00000-05000 12 777
# bash ./scripts/NATS-Bench/train-topology.sh 05001-10000 12 777
# bash ./scripts/NATS-Bench/train-topology.sh 10001-14500 12 777
# bash ./scripts/NATS-Bench/train-topology.sh 14501-15624 12 777
#
##############################################################################
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

save_dir=./output/NATS-Bench-topology/

OMP_NUM_THREADS=${cpus} python exps/NATS-Bench/main-tss.py \
	--mode new --srange ${srange} --hyper ${opt} --save_dir ${save_dir} \
	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
	--splits   1       0       0        0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python/ImageNet16 \
	--workers ${cpus} \
	--seeds ${all_seeds}
