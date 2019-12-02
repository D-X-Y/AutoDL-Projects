#!/bin/bash
# bash ./scripts-search/AA-NAS-train-archs.sh 0 100 -1 '777 888 999'
echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for start and end and arch-index"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

xstart=$1
xend=$2
arch_index=$3
all_seeds=$4

if [ ${arch_index} == "-1" ]; then
  mode=new
else
  mode=cover
fi

save_dir=./output/AA-NAS-BENCH-4/

OMP_NUM_THREADS=4 python ./exps/AA-NAS-Bench-main.py \
	--mode ${mode} --save_dir ${save_dir} --max_node 4 \
	--use_less 0 \
	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
	--splits   1       0       0        0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python/ImageNet16 \
	--channel 16 --num_cells 5 \
	--workers 4 \
	--srange ${xstart} ${xend} --arch_index ${arch_index} \
	--seeds ${all_seeds}
