#!/bin/bash
# bash ./scripts/com-paddle.sh cifar10
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for the dataset and the-model-name and the-random-seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1

PY_C="./env/bin/python"
if [ ! -f ${PY_C} ]; then
  echo "Local Run with Python: "`which python`
  PY_C="python"
  SAVE_ROOT="./output/com-paddle"
else
  echo "Cluster Run with Python: "${PY_C}
  SAVE_ROOT="./hadoop-data/COM-PADDLE"
fi

basic_func(){
  dataset=$1
  model=$2
  batch=$3
  rseed=$4
  save_dir=${SAVE_ROOT}/${dataset}-${model}-${batch}
  ${PY_C}  ./exps/basic-main.py --dataset ${dataset} \
	--data_path $TORCH_HOME/cifar.python \
	--model_config ./configs/archs/CIFAR-${model}.config \
	--optim_config ./configs/opts/Com-Paddle-RES.config \
	--procedure    basic \
	--save_dir ${save_dir} --cutout_length -1 --batch_size  ${batch} --rand_seed ${rseed} --workers 4 --eval_frequency 1 --print_freq 100 --print_freq_eval 200
}

nas_infer_func(){
  dataset=$1
  model=$2
  batch=$3
  rseed=$4
  save_dir=${SAVE_ROOT}/${dataset}-${model}-${batch}
  ${PY_C}  ./exps/basic-main.py --dataset ${dataset} \
	--data_path $TORCH_HOME/cifar.python --model_source nas \
	--model_config ./configs/archs/NAS-CIFAR-${model}.config \
	--optim_config ./configs/opts/Com-Paddle-NAS.config \
	--procedure    basic \
	--save_dir ${save_dir} --cutout_length -1 --batch_size  ${batch} --rand_seed ${rseed} --workers 4 --eval_frequency 1 --print_freq 100 --print_freq_eval 200
}

#datasets="cifar10 cifar100"

#datasets="cifar10 cifar100"
#for dataset in ${datasets}
#do
#basic_func ${dataset} ResNet20  256 -1
#basic_func ${dataset} ResNet32  256 -1
#basic_func ${dataset} ResNet110 256 -1
#done

nas_infer_func ${dataset} GDAS_V1 96 -1
nas_infer_func ${dataset} SETN    96 -1
