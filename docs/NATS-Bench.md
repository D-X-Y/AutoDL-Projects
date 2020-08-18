# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size

Neural architecture search (NAS) has attracted a lot of attention and has been illustrated to bring tangible benefits in a large number of applications in the past few years. Network topology and network size have been regarded as two of the most important aspects for the performance of deep learning models and the community has spawned lots of searching algorithms for both of those aspects of the neural architectures. However, the performance gain from these searching algorithms is achieved under different search spaces and training setups. This makes the overall performance of the algorithms incomparable and the improvement from a sub-module of the searching model unclear.
In this paper, we propose NATS-Bench, a unified benchmark on searching for both topology and size, for (almost) any up-to-date NAS algorithm.
NATS-Bench includes the search space of 15,625 neural cell candidates for architecture topology and 32,768 for architecture size on three datasets.
We analyze the validity of our benchmark in terms of various criteria and performance comparison of all candidates in the search space.
We also show the versatility of NATS-Bench by benchmarking 13 recent state-of-the-art NAS algorithms on it. All logs and diagnostic information trained using the same setup for each candidate are provided.
This facilitates a much larger community of researchers to focus on developing better NAS algorithms in a more comparable and computationally effective environment.


**coming soon!**


## How to Use NATS-Bench


## The Procedure of Creating NATS-Bench

1, train all architecture candidate in the size search space with 90 epochs and use the random seed of `777`.
```
bash ./scripts/NATS-Bench/train-shapes.sh 00000-32767 90 777
```
The checkpoint of all candidates are located at `output/NATS-Bench-size` by default



## To Reproduce 13 Baseline NAS Algorithms in NAS-Bench-201

### Reproduce NAS methods on the topology search space

```
DARTS (V1):
python ./exps/algos-v2/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1
python ./exps/algos-v2/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v1
python ./exps/algos-v2/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v1

DARTS (V2):
python ./exps/algos-v2/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v2
python ./exps/algos-v2/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v2
python ./exps/algos-v2/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v2

GDAS:
python ./exps/algos-v2/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas
python ./exps/algos-v2/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo gdas
python ./exps/algos-v2/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16

SETN:
python ./exps/algos-v2/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo setn
python ./exps/algos-v2/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo setn
python ./exps/algos-v2/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo setn

Random Search with Weight Sharing:
python ./exps/algos-v2/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random
python ./exps/algos-v2/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo random
python ./exps/algos-v2/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo random

ENAS:
python ./exps/algos-v2/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001
python ./exps/algos-v2/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001
python ./exps/algos-v2/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001
```

### Reproduce NAS methods on the size search space


### Final Discovered Architectures for Each Algorithm

The architecture index can be found by use `api.query_index_by_arch(architecture_string)`.

The final discovered architecture ID on CIFAR-10:
```
DARTS (V1):
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|

DARTS (V2):
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|

GDAS:
|nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|
|nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
|avg_pool_3x3~0|+|nor_conv_3x3~0|skip_connect~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_1x1~2|
```

The final discovered architecture ID on CIFAR-100:
```
DARTS (V1):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|none~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|none~2|
|skip_connect~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|

DARTS (V2):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|skip_connect~2|
|skip_connect~0|+|nor_conv_3x3~0|none~1|+|skip_connect~0|none~1|none~2|
|skip_connect~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|skip_connect~1|none~2|

GDAS:
|nor_conv_3x3~0|+|nor_conv_1x1~0|none~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
|avg_pool_3x3~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|avg_pool_3x3~1|nor_conv_1x1~2|
|avg_pool_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_1x1~2|
```

The final discovered architecture ID on ImageNet-16-120:
```
DARTS (V1):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_1x1~2|

DARTS (V2):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|skip_connect~2|

GDAS:
|none~0|+|none~0|none~1|+|nor_conv_3x3~0|none~1|none~2|
|none~0|+|none~0|none~1|+|nor_conv_3x3~0|none~1|none~2|
|none~0|+|none~0|none~1|+|nor_conv_3x3~0|none~1|none~2|
```
