#!/bin/bash
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for the architectures"
  exit 1               
fi 

arch=$1
SAVED=./output/NAS-RNN/Search-${arch}-WT2
PY_C="./env/bin/python"

if [ ! -f ${PY_C} ]; then
  echo "Local Run with Python: "`which python`
  PY_C="python"
else
  echo "Cluster Run with Python: "${PY_C}
fi

${PY_C} --version

${PY_C} ./exps-rnn/train_rnn_base.py \
	--arch ${arch} \
	--save_path ${SAVED} \
	--config_path ./configs/NAS-WT2-BASE.config \
	--print_freq 300
