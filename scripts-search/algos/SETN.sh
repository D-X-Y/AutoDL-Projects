#!/bin/bash
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019
# bash ./scripts-search/algos/SETN.sh cifar10 0 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, BN-tracking-status, and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
BN=$2
seed=$3
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth
benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/search-cell-${space}/SETN-${dataset}-BN${BN}

OMP_NUM_THREADS=4 python ./exps/algos/SETN.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--config_path configs/nas-benchmark/algos/SETN.config \
	--track_running_stats ${BN} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--select_num 100 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
