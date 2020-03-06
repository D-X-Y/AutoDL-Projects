#!/bin/bash
# bash ./scripts/retrain-searched-net.sh cifar10 ${NAME} ${PATH} 256 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for dataset, the save dir base name, the model path, the batch size, the random seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
save_name=$2
model_path=$3
batch=$4
rseed=$5

if [ ${dataset} == 'cifar10' ] || [ ${dataset} == 'cifar100' ]; then
  xpath=$TORCH_HOME/cifar.python
  base=CIFAR
  workers=4
  cutout_length=16
elif [ ${dataset} == 'imagenet-1k' ]; then
  xpath=$TORCH_HOME/ILSVRC2012
  base=IMAGENET
  workers=28
  cutout_length=-1
else
  exit 1
  echo 'Unknown dataset: '${dataset}
fi

SAVE_ROOT="./output"

save_dir=${SAVE_ROOT}/nas-infer/${dataset}-BS${batch}-${save_name}

python --version

python ./exps/basic-main.py --dataset ${dataset} \
	--data_path ${xpath} --model_source autodl-searched \
	--model_config ./configs/archs/NAS-${base}-none.config \
	--optim_config ./configs/opts/NAS-${base}.config \
	--extra_model_path ${model_path} \
	--procedure    basic \
	--save_dir     ${save_dir} \
	--cutout_length ${cutout_length} \
	--batch_size  ${batch} --rand_seed ${rseed} --workers ${workers} \
	--eval_frequency 1 --print_freq 500 --print_freq_eval 1000
