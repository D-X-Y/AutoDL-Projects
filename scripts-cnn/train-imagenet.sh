#!/usr/bin/env sh
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for the architecture, and the channel and the layers"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

arch=$1
dataset=imagenet
channels=$2
layers=$3
SAVED=./output/NAS-CNN/${arch}-${dataset}-C${channels}-L${layers}-E250

PY_C="./env/bin/python"

if [ ! -f ${PY_C} ]; then
  echo "Local Run with Python: "`which python`
  PY_C="python"
else
  echo "Cluster Run with Python: "${PY_C}
  echo "Unzip ILSVRC2012"
  tar xf ./hadoop-data/ILSVRC2012.tar   -C ${TORCH_HOME}
fi

${PY_C} --version

${PY_C} ./exps-cnn/train_base.py \
	--data_path $TORCH_HOME/ILSVRC2012 \
	--dataset ${dataset} --arch ${arch} \
	--save_path ${SAVED} \
	--grad_clip 5 \
	--init_channels ${channels} --layers ${layers} \
	--model_config ./configs/nas-imagenet.config \
	--print_freq 200 --workers 20
