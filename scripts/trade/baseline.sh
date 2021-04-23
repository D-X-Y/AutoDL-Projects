#!/bin/bash
#
# bash scripts/trade/baseline.sh 0 csi300
# bash scripts/trade/baseline.sh 1 csi100
# bash scripts/trade/baseline.sh 1 all
#
set -e
echo script name: $0
echo $# arguments

if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  exit 1
fi

gpu=$1
market=$2

# algorithms="NAIVE-V1 NAIVE-V2 MLP GRU LSTM ALSTM XGBoost LightGBM SFM TabNet DoubleE"
algorithms="XGBoost LightGBM SFM TabNet DoubleE"

for alg in ${algorithms}
do
  python exps/trading/baselines.py --alg ${alg} --gpu ${gpu} --market ${market}
done
