#!/bin/bash
#
# bash scripts/trade/tsf-time.sh 0 csi300 TSF-2x24-drop0_0
# bash scripts/trade/tsf-time.sh 1 csi100
# bash scripts/trade/tsf-time.sh 1 all
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
base=$3
xtimes="2008-01-01 2008-07-01 2009-01-01 2009-07-01 2010-01-01 2011-01-01 2012-01-01 2013-01-01"

for xtime in ${xtimes}
do

  python exps/trading/baselines.py --alg ${base}s${xtime} --gpu ${gpu} --market ${market} --shared_dataset False

done
