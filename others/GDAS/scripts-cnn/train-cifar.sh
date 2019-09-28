#!/usr/bin/env sh
# bash scripts-cnn/train-cifar.sh GDAS cifar10 cut
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for the architecture, and the dataset-name, and the cutout"
  exit 1               
fi 
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

arch=$1
dataset=$2
cutout=$3
SAVED=./output/NAS-CNN/${arch}-${dataset}-${cutout}-E600

PY_C="./env/bin/python"

if [ ! -f ${PY_C} ]; then
  echo "Local Run with Python: "`which python`
  PY_C="python"
else
  echo "Cluster Run with Python: "${PY_C}
fi

${PY_C} --version

${PY_C} ./exps-cnn/train_base.py \
        --data_path $TORCH_HOME/cifar.python \
	--dataset ${dataset} --arch ${arch} \
	--save_path ${SAVED} \
	--grad_clip 5 \
	--init_channels 36 --layers 20 \
	--model_config ./configs/nas-cifar-cos-${cutout}.config \
	--print_freq 100 --workers 6
