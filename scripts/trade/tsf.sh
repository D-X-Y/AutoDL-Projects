#!/bin/bash
#
# bash scripts/trade/tsf.sh 0 csi300 3
# bash scripts/trade/tsf.sh 1 csi100 3
# bash scripts/trade/tsf.sh 1 all    3
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
depth=$3

channels="6 12 24 32 48 64"

for channel in ${channels}
do

  python exps/trading/baselines.py --alg TSF-${depth}x${channel}-d0 --gpu ${gpu} --market ${market}

done
