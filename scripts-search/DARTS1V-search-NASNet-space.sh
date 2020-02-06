#!/bin/bash
# bash ./scripts-search/DARTS1V-search-NASNet-space.sh cifar10 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for dataset, and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
BN=1
seed=$2
channel=16
num_cells=5
max_nodes=4
space=darts

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/search-cell-${space}/DARTS-V1-${dataset}-BN${BN}

OMP_NUM_THREADS=4 python ./exps/algos/DARTS-V1.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path configs/search-opts/DARTS-NASNet-CIFAR.config \
	--model_config configs/search-archs/DARTS-NASNet-CIFAR.config \
	--track_running_stats ${BN} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
