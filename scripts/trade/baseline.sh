#!/bin/bash
# bash scripts/trade/baseline.sh 0 csi300
set -e
echo script name: $0
echo $# arguments

if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  exit 1
fi

gpu=$1
market=$2

algorithms="MLP GRU LSTM ALSTM XGBoost LightGBM"

for alg in ${algorithms}
do

  python exps/trading/baselines.py --alg ${alg} --gpu ${gpu} --market ${market}

done
