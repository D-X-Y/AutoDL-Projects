#!/bin/bash
# bash ./exps/algos-v2/run-all.sh
set -e
echo script name: $0
echo $# arguments

datasets="cifar10 cifar100 ImageNet16-120"
search_spaces="tss sss"

for dataset in ${datasets}
do
  for search_space in ${search_spaces}
  do
    python ./exps/algos-v2/reinforce.py --dataset ${dataset} --search_space ${search_space} --learning_rate 0.001
    python ./exps/algos-v2/regularized_ea.py --dataset ${dataset} --search_space ${search_space} --ea_cycles 200 --ea_population 10 --ea_sample_size 3
    python ./exps/algos-v2/random_wo_share.py --dataset ${dataset} --search_space ${search_space}
    python exps/algos-v2/bohb.py --dataset ${dataset} --search_space ${search_space} --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3
  done
done
