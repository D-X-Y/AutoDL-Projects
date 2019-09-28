#!/usr/bin/env sh
# bash ./scripts/base-imagenet.sh ResNet110 Step-Soft 256
set -e
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for the architecture, and softmax/smooth-softmax, and batch-size, and seed"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

arch=$1
opt=$2
batch=$3
rand_seed=$4
dataset=imagenet-1k

PY_C="./env/bin/python"
if [ ! -f ${PY_C} ]; then
  echo "Local Run with Python: "`which python`
  PY_C="python"
  SAVE_ROOT="./output"
else
  echo "Cluster Run with Python: "${PY_C}
  SAVE_ROOT="./hadoop-data/SearchCheckpoints"
  echo "Unzip ILSVRC2012"
  tar --version
  tar -xf ./hadoop-data/ILSVRC2012.tar -C ${TORCH_HOME}
  echo "Unzip ILSVRC2012 done"
fi

if [ ${opt} = "RMSProp" ]; then
  epoch=E200
elif [ ${opt} = "Shuffle" ]; then
  epoch=E240
  dataset=imagenet-1k
elif [ ${opt} = "MobileS" ]; then
  epoch=E480
  dataset=imagenet-1k-s
elif [ ${opt} = "MobileFast" ] || [ ${opt} = "MobileFastS" ]; then
  epoch=E150
  dataset=imagenet-1k-s
else
  epoch=E120
fi

if [ ${batch} = "256" ]; then
  opt_dir=opts
  workers=24
elif [ ${batch} = "1024" ]; then
  opt_dir=opts-1K
  workers=48
else
  echo "Invalid batch size : "${batch}
  exit 1
fi

save_dir=${SAVE_ROOT}/basic/${dataset}/${arch}-${opt}-${epoch}-${batch}

${PY_C} --version

${PY_C} ./exps/basic-main.py --dataset ${dataset} \
	--data_path $TORCH_HOME/ILSVRC2012 \
	--model_config ./configs/archs/ImageNet-${arch}.config \
	--optim_config ./configs/${opt_dir}/ImageNet-${epoch}-${opt}.config \
	--procedure    basic \
	--save_dir     ${save_dir} \
	--cutout_length -1 \
	--batch_size  ${batch} --rand_seed ${rand_seed} --workers ${workers} \
	--eval_frequency 1 --print_freq 500 --print_freq_eval 2000
