#!/bin/bash
# bash ./scripts-search/algos/REINFORCE.sh -1
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

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/cell-search-tiny/REINFORCE-${dataset}

OMP_NUM_THREADS=4 python ./exps/algos/reinforce.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name aa-nas \
	--arch_nas_dataset ./output/AA-NAS-BENCH-4/simplifies/C16-N5-final-infos.pth \
	--learning_rate 0.001 --RL_steps 100 --EMA_momentum 0.9 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
