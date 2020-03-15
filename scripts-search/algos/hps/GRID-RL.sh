#!/bin/bash
echo script name: $0

#lrs="0.01 0.02 0.1 0.2 0.5 1.0 1.5 2.0 2.5 3.0"
lrs="0.01 0.02 0.1 0.2 0.5"

for lr in ${lrs}
do
  bash ./scripts-search/algos/REINFORCE.sh ${lr} -1
done
