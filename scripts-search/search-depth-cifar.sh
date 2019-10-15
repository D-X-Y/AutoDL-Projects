#!/bin/bash
# bash ./scripts-search/search-depth-cifar.sh cifar10 ResNet110 CIFAR 0 0 0.57 777
# bash ./scripts-search/search-depth-cifar.sh cifar10 ResNet110 CIFARX 0 0 0.57 777
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 7 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 7 parameters for the dataset and the-model-name and the-optimizer and gumbel-max/min and FLOP-ratio and the-random-seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
model=$2
optim=$3
batch=256
gumbel_min=$4
gumbel_max=$5
expected_FLOP_ratio=$6
rseed=$7


SAVE_ROOT="./output"

save_dir=${SAVE_ROOT}/search-depth/${dataset}-${model}-${optim}-Gumbel_${gumbel_min}_${gumbel_max}-${expected_FLOP_ratio}

python --version

OMP_NUM_THREADS=4 python ./exps/search-shape.py --dataset ${dataset} \
	--data_path $TORCH_HOME/cifar.python \
	--model_config ./configs/archs/CIFAR-${model}.config \
	--split_path   ./.latent-data/splits/${dataset}-0.5.pth \
	--optim_config ./configs/search-opts/${optim}.config \
	--search_shape   depth \
	--procedure      search \
	--FLOP_ratio     ${expected_FLOP_ratio} \
	--FLOP_weight    2 --FLOP_tolerant 0.05 \
	--save_dir       ${save_dir} \
	--gumbel_tau_max ${gumbel_max} --gumbel_tau_min ${gumbel_min} \
	--cutout_length -1 \
	--batch_size  ${batch} --rand_seed ${rseed} --workers 4 \
	--eval_frequency 1 --print_freq 100 --print_freq_eval 200


if [ "$rseed" = "-1" ]; then
  echo "Skip training the best configuration"
else
  # normal training
  xsave_dir=${save_dir}/seed-${rseed}-NMT
  OMP_NUM_THREADS=4 python ./exps/basic-main.py --dataset ${dataset} \
	--data_path $TORCH_HOME/cifar.python \
	--model_config ${save_dir}/seed-${rseed}-last.config \
	--optim_config ./configs/opts/CIFAR-E300-W5-L1-COS.config \
	--procedure    basic \
	--save_dir     ${xsave_dir} \
	--cutout_length -1 \
	--batch_size 256 --rand_seed ${rseed} --workers 4 \
	--eval_frequency 1 --print_freq 100 --print_freq_eval 200
  # KD training
  xsave_dir=${save_dir}/seed-${rseed}-KDT
  OMP_NUM_THREADS=4 python ./exps/KD-main.py --dataset ${dataset} \
	--data_path $TORCH_HOME/cifar.python \
	--model_config  ${save_dir}/seed-${rseed}-last.config \
	--optim_config  ./configs/opts/CIFAR-E300-W5-L1-COS.config \
	--KD_checkpoint ./.latent-data/basemodels/${dataset}/${model}.pth \
	--procedure    Simple-KD \
	--save_dir     ${xsave_dir} \
	--KD_alpha 0.9 --KD_temperature 4 \
	--cutout_length -1 \
	--batch_size 256 --rand_seed ${rseed} --workers 4 \
	--eval_frequency 1 --print_freq 100 --print_freq_eval 200
fi
