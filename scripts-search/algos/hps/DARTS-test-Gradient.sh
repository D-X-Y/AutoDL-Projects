#!/bin/bash
# bash ./scripts-search/algos/DARTS-test-Gradient.sh cifar10 0 5
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, tracking_status, and gradient_clip"
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
gradient_clip=$3
seed=-1
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

save_dir=./output/search-cell-${space}/DARTS-V1-${dataset}-BN${BN}-Gradient${gradient_clip}

OMP_NUM_THREADS=4 python ./exps/algos/DARTS-V1.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path configs/nas-benchmark/algos/DARTS.config \
	--arch_nas_dataset ${benchmark_file} \
	--track_running_stats ${BN} --gradient_clip ${gradient_clip} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
