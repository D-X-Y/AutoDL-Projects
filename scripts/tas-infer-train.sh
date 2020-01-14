#!/bin/bash
# bash ./scripts/tas-infer-train.sh cifar10 C100-ResNet32 -1
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for the dataset and the-config-name and the-random-seed"
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
rseed=$3
batch=256

save_dir=./output/search-shape/TAS-INFER-${dataset}-${model}

if [ ${dataset} == 'cifar10' ] || [ ${dataset} == 'cifar100' ]; then
  xpath=$TORCH_HOME/cifar.python
  opt_config=./configs/opts/CIFAR-E300-W5-L1-COS.config
  workers=4
elif [ ${dataset} == 'imagenet-1k' ]; then
  xpath=$TORCH_HOME/ILSVRC2012
  #opt_config=./configs/opts/ImageNet-E120-Cos-Smooth.config
  opt_config=./configs/opts/RImageNet-E120-Cos-Soft.config
  workers=28
else
  echo 'Unknown dataset: '${dataset}
  exit 1
fi

python --version

# normal training
xsave_dir=${save_dir}-NMT
OMP_NUM_THREADS=4 python ./exps/basic-main.py --dataset ${dataset} \
	--data_path ${xpath} \
	--model_config ./configs/NeurIPS-2019/${model}.config \
	--optim_config ${opt_config} \
	--procedure    basic \
	--save_dir     ${xsave_dir} \
	--cutout_length -1 \
	--batch_size ${batch} --rand_seed ${rseed} --workers ${workers} \
	--eval_frequency 1 --print_freq 100 --print_freq_eval 200

# KD training
xsave_dir=${save_dir}-KDT
OMP_NUM_THREADS=4 python ./exps/KD-main.py --dataset ${dataset} \
	--data_path ${xpath} \
	--model_config ./configs/NeurIPS-2019/${model}.config \
	--optim_config  ${opt_config} \
	--KD_checkpoint ./.latent-data/basemodels/${dataset}/${model}.pth \
	--procedure    Simple-KD \
	--save_dir     ${xsave_dir} \
	--KD_alpha 0.9 --KD_temperature 4 \
	--cutout_length -1 \
	--batch_size ${batch} --rand_seed ${rseed} --workers ${workers} \
	--eval_frequency 1 --print_freq 100 --print_freq_eval 200
