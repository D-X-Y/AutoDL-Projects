#!/bin/bash
# bash ./NATS/search-size.sh 0 777
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for GPU-device and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

device=$1
seed=$2

CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo tas --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo tas --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo tas --rand_seed ${seed}

CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo fbv2 --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo fbv2 --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo fbv2 --rand_seed ${seed}

CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo tunas --arch_weight_decay 0 --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo tunas --arch_weight_decay 0 --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo tunas --arch_weight_decay 0 --rand_seed ${seed}
