#!/bin/bash
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01                          #
##############################################################################
# CUDA_VISIBLE_DEVICES=0 bash scripts/NATS-Bench/train-shapes.sh 00000-05000 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 05001-10000 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 10001-14500 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 14501-18000 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 18001-19500 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 19501-23500 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 23501-27500 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 27501-30000 12 777
# bash ./scripts/NATS-Bench/train-shapes.sh 30001-32767 12 777
#
# CUDA_VISIBLE_DEVICES=2 bash ./scripts/NATS-Bench/train-shapes.sh 01000-03999,04050-05000,06000-09000,11000-14500,15000-18500,20000-23500,25000-27500,29000-30000 12 777
# SLURM_PROCID=1 SLURM_NTASKS=5 bash ./scripts/NATS-Bench/train-shapes.sh 01000-03999,04050-05000,06000-09000,11000-14500,15000-18500,20000-23500,25000-27500,29000-30000 90 777
# [GCP] bash ./scripts/NATS-Bench/train-shapes.sh 00000-09999 90 777
# [UTS] bash ./scripts/NATS-Bench/train-shapes.sh 30000-32767 90 777
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

save_dir=./output/NATS-Bench-size/

OMP_NUM_THREADS=${cpus} python exps/NATS-Bench/main-sss.py \
	--mode new --srange ${srange} --hyper ${opt} --save_dir ${save_dir} \
	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
	--splits   1       0       0        0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python/ImageNet16 \
	--workers ${cpus} \
	--seeds ${all_seeds}
