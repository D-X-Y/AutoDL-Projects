# [Searching for A Robust Neural Architecture in Four GPU Hours](https://arxiv.org/abs/1910.04465)

<img align="right" src="https://d-x-y.github.com/resources/paper-icon/CVPR-2019-GDAS.png" width="300">

Searching for A Robust Neural Architecture in Four GPU Hours is accepted at CVPR 2019.
In this paper, we proposed a Gradient-based searching algorithm using Differentiable Architecture Sampling (GDAS).
GDAS is baseed on DARTS and improves it with Gumbel-softmax sampling.
Concurrently at the submission period, several NAS papers (SNAS and FBNet) also utilized Gumbel-softmax sampling. We are different at how to forward and backward, see more details in our paper and codes.
Experiments on CIFAR-10, CIFAR-100, ImageNet, PTB, and WT2 are reported.


## Requirements and Preparation

Please install `Python>=3.6` and `PyTorch>=1.2.0`.

CIFAR and ImageNet should be downloaded and extracted into `$TORCH_HOME`.

### Usefull tools
1. Compute the number of parameters and FLOPs of a model:
```
from utils import get_model_infos
flop, param  = get_model_infos(net, (1,3,32,32))
```

2. Different NAS-searched architectures are defined [here](https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/nas_infer_model/DXYs/genotypes.py).


## Usage

### Reproducing the results of our searched architecture in GDAS
Please use the following scripts to train the searched GDAS-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k GDAS_V1 256 -1
```
If you are interested in the configs of each NAS-searched architecture, they are defined at [genotypes.py](https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/nas_infer_model/DXYs/genotypes.py).

### Searching on the NASNet search space

Please use the following scripts to use GDAS to search as in the original paper:
```
# search for both normal and reduction cells
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NASNet-space-search-by-GDAS.sh cifar10 1 -1

# search for the normal cell while use a fixed reduction cell
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NASNet-space-search-by-GDAS-FRC.sh cifar10 1 -1
```

**After searching**, if you want to re-train the searched architecture found by the above script, you can use the following script:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/retrain-searched-net.sh cifar10 gdas-searched \
		     output/search-cell-darts/GDAS-cifar10-BN1/checkpoint/seed-945-basic.pth 96 -1
```
Note that `gdas-searched` is a string to indicate the name of the saved dir and `output/search-cell-darts/GDAS-cifar10-BN1/checkpoint/seed-945-basic.pth` is the file path that the searching algorithm generated.

The above script does not apply heavy augmentation to train the model, so the accuracy will be lower than the original paper.
If you want to change the default hyper-parameter for re-training, please have a look at `./scripts/retrain-searched-net.sh` and `configs/archs/NAS-*-none.config`.


### Searching on a small search space (NAS-Bench-201)

The GDAS searching codes on a small search space:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh cifar10 1 -1
```

The baseline searching codes are DARTS:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V1.sh cifar10 1 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V2.sh cifar10 1 -1
```

**After searching**, if you want to train the searched architecture found by the above scripts, please use the following codes:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NAS-Bench-201/train-a-net.sh '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|' 16 5
```
`|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|` represents the structure of a searched architecture. My codes will automatically print it during the searching procedure.


**Tensorflow codes for GDAS are in experimental state**, which locates at `exps-tf`.

# Citation

If you find that this project helps your research, please consider citing the following paper:
```
@inproceedings{dong2019search,
  title     = {Searching for A Robust Neural Architecture in Four GPU Hours},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1761--1770},
  year      = {2019}
}
```
