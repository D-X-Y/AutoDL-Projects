#!/bin/bash
# bash ./scripts-search/AA-NAS-meta-gen.sh AA-NAS-BENCHMARK 4
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for save-dir-name and maximum-node-in-cell"
  exit 1
fi

name=$1
node=$2

save_dir=./output/${name}-${node}

python ./exps/AA-NAS-Bench-main.py --mode meta --save_dir ${save_dir} --max_node ${node}
