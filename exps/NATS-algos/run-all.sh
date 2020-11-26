#!/bin/bash
# bash ./exps/NATS-algos/run-all.sh mul
# bash ./exps/NATS-algos/run-all.sh ws
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for type of algorithms."
  exit 1
fi

alg_type=$1

if [ "$alg_type" == "mul" ]; then
  # datasets="cifar10 cifar100 ImageNet16-120"
  run_four_algorithms(){
    dataset=$1
    search_space=$2
    time_budget=$3
    python ./exps/NATS-algos/reinforce.py       --dataset ${dataset} --search_space ${search_space} --time_budget ${time_budget} --learning_rate 0.01
    python ./exps/NATS-algos/regularized_ea.py  --dataset ${dataset} --search_space ${search_space} --time_budget ${time_budget} --ea_cycles 200 --ea_population 10 --ea_sample_size 3
    python ./exps/NATS-algos/random_wo_share.py --dataset ${dataset} --search_space ${search_space} --time_budget ${time_budget}
    python ./exps/NATS-algos/bohb.py            --dataset ${dataset} --search_space ${search_space} --time_budget ${time_budget} --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3
  }
  # The topology search space
  run_four_algorithms "cifar10"        "tss" "20000"
  run_four_algorithms "cifar100"       "tss" "40000"
  run_four_algorithms "ImageNet16-120" "tss" "120000"

  # The size search space
  run_four_algorithms "cifar10"        "sss" "20000"
  run_four_algorithms "cifar100"       "sss" "40000"
  run_four_algorithms "ImageNet16-120" "sss" "60000"
  # python exps/experimental/vis-bench-algos.py --search_space tss
  # python exps/experimental/vis-bench-algos.py --search_space sss
else
  seeds="777 888 999"
  algos="darts-v1 darts-v2 gdas setn random enas"
  epoch=200
  for seed in ${seeds}
  do
    for alg in ${algos}
    do
      python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo ${alg} --rand_seed ${seed} --overwite_epochs ${epoch}
      python ./exps/NATS-algos/search-cell.py --dataset cifar100  --data_path $TORCH_HOME/cifar.python --algo ${alg} --rand_seed ${seed} --overwite_epochs ${epoch}
      python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120  --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo ${alg} --rand_seed ${seed} --overwite_epochs ${epoch}
    done
  done
fi

