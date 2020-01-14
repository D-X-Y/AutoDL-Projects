#!/bin/bash
# bash ./scripts-search/algos/REINFORCE.sh 0.001 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for LR and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=cifar10
LR=$1
seed=$2
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

save_dir=./output/search-cell-${space}/REINFORCE-${dataset}-${LR}

OMP_NUM_THREADS=4 python ./exps/algos/reinforce.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} \
	--search_space_name ${space} \
	--arch_nas_dataset ${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth \
	--time_budget 12000 \
	--learning_rate ${LR} --EMA_momentum 0.9 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
