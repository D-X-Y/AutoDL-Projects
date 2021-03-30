#!/bin/bash
#
# bash scripts/trade/tsf-all.sh 0 csi300 0_0
# bash scripts/trade/tsf-all.sh 0 csi300 0.1_0
# bash scripts/trade/tsf-all.sh 1 all    
#
set -e
echo script name: $0
echo $# arguments

if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  exit 1
fi

gpu=$1
market=$2
drop=$3

channels="6 12 24 32 48 64"
depths="1 2 3 4 5 6"

for channel in ${channels}
do
  for depth in ${depths}
  do
    python exps/trading/baselines.py --alg TSF-${depth}x${channel}-drop${drop} --gpu ${gpu} --market ${market}
  done
done
