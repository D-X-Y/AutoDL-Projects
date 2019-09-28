#!/bin/bash
# bash ./scripts/KD-train.sh cifar10 ResNet110 ResNet110 0.5 1 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 6 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 6 parameters for the dataset / the-model-name / the-teacher-path / KD-alpha / KD-temperature / the-random-seed"
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
teacher=$3
alpha=$4
temperature=$5
epoch=E300
LR=L1
batch=256
rseed=$6

save_dir=./output/KD/${dataset}-${teacher}.2.${model}-${alpha}-${temperature}
rm -rf ${save_dir}

PY_C="./env/bin/python"
if [ ! -f ${PY_C} ]; then
  echo "Local Run with Python: "`which python`
  PY_C="python"
else
  echo "Cluster Run with Python: "${PY_C}
fi

${PY_C} --version

${PY_C} ./exps/KD-main.py --dataset ${dataset} \
	--data_path $TORCH_HOME/cifar.python \
	--model_config  ./configs/archs/CIFAR-${model}.config \
	--optim_config  ./configs/opts/CIFAR-${epoch}-W5-${LR}-COS.config \
	--KD_checkpoint $TORCH_HOME/TAS-checkpoints/basemodels/${dataset}/${teacher}.pth \
	--procedure    Simple-KD \
	--save_dir     ${save_dir} \
	--KD_alpha ${alpha} --KD_temperature ${temperature} \
	--cutout_length -1 \
	--batch_size  ${batch} --rand_seed ${rseed} --workers 4 \
	--eval_frequency 1 --print_freq 100 --print_freq_eval 200
