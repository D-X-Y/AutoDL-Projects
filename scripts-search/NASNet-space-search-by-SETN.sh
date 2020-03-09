#!/bin/bash
# bash ./scripts-search/NASNet-space-search-by-SETN.sh cifar10 1 -1
# [TO BE DONE]
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, track_running_stats, and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
track_running_stats=$2
seed=$3
space=darts

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/search-cell-${space}/SETN-${dataset}-BN${track_running_stats}

OMP_NUM_THREADS=4 python ./exps/algos/SETN.py \
	--save_dir ${save_dir} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path  configs/search-opts/SETN-NASNet-CIFAR.config \
	--model_config configs/search-archs/SETN-NASNet-CIFAR.config \
	--track_running_stats ${track_running_stats} \
	--select_num 1000 \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
