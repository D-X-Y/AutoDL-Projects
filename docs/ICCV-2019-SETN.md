# [One-Shot Neural Architecture Search via Self-Evaluated Template Network](https://arxiv.org/abs/1910.05733)

<img align="right" src="https://d-x-y.github.com/resources/paper-icon/ICCV-2019-SETN.png" width="450">

<strong>Highlight</strong>: we equip one-shot NAS with an architecture sampler and train network weights using uniformly sampling.

One-Shot Neural Architecture Search via Self-Evaluated Template Network is accepted by ICCV 2019.


## Requirements and Preparation

Please install `Python>=3.6` and `PyTorch>=1.2.0`.

### Usefull tools
1. Compute the number of parameters and FLOPs of a model:
```
from utils import get_model_infos
flop, param  = get_model_infos(net, (1,3,32,32))
```

2. Different NAS-searched architectures are defined [here](https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/nas_infer_model/DXYs/genotypes.py).


## Usage

Please use the following scripts to train the searched SETN-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  SETN 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 SETN 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k SETN  256 -1
```

The searching codes of SETN on a small search space (NAS-Bench-201).
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/SETN.sh cifar10 -1
```


# Citation

If you find that this project helps your research, please consider citing the following paper:
```
@inproceedings{dong2019one,
  title     = {One-Shot Neural Architecture Search via Self-Evaluated Template Network},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {3681--3690},
  year      = {2019}
}
```
