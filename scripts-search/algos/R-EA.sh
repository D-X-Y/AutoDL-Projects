#!/bin/bash
# Regularized Evolution for Image Classifier Architecture Search, AAAI 2019
# bash ./scripts-search/algos/R-EA.sh -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=cifar10
seed=$1
channel=16
num_cells=5
max_nodes=4
space=nas-bench-102

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/search-cell-${space}/R-EA-${dataset}

OMP_NUM_THREADS=4 python ./exps/algos/R_EA.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--arch_nas_dataset ${TORCH_HOME}/NAS-Bench-102-v1_0-e61699.pth \
	--time_budget 12000 \
	--ea_cycles 100 --ea_population 10 --ea_sample_size 3 --ea_fast_by_api 1 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
