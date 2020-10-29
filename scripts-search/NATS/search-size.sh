#!/bin/bash
# bash scripts-search/NATS/search-size.sh 0 0.3 777
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for GPU-device, warmup-ratio, and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

device=$1
ratio=$2
seed=$3

CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo tas --warmup_ratio ${ratio} --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo tas --warmup_ratio ${ratio} --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo tas --warmup_ratio ${ratio} --rand_seed ${seed}

#
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo mask_gumbel --warmup_ratio ${ratio} --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo mask_gumbel --warmup_ratio ${ratio} --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo mask_gumbel --warmup_ratio ${ratio} --rand_seed ${seed}

#
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo mask_rl --arch_weight_decay 0 --warmup_ratio ${ratio} --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo mask_rl --arch_weight_decay 0 --warmup_ratio ${ratio} --rand_seed ${seed}
CUDA_VISIBLE_DEVICES=${device} python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo mask_rl --arch_weight_decay 0 --warmup_ratio ${ratio} --rand_seed ${seed}
