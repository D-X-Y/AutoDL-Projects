# Neural Architecture Search (NAS)

This project contains the following neural architecture search (NAS) algorithms, implemented in [PyTorch](http://pytorch.org).
More NAS resources can be found in [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS).

- NAS-Bench-102: Extending the Scope of Reproducible Neural Architecture Search, ICLR 2020
- Network Pruning via Transformable Architecture Search, NeurIPS 2019
- One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019
- Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019
- 10 NAS algorithms for the neural topology in `exps/algos` (see [NAS-Bench-102.md](https://github.com/D-X-Y/NAS-Projects/blob/master/NAS-Bench-102.md) for more details)
- Several typical classification models, e.g., ResNet and DenseNet (see [BASELINE.md](https://github.com/D-X-Y/NAS-Projects/blob/master/BASELINE.md))


## Requirements and Preparation

Please install `PyTorch>=1.2.0`, `Python>=3.6`, and `opencv`.

CIFAR and ImageNet should be downloaded and extracted into `$TORCH_HOME`.
Some methods use knowledge distillation (KD), which require pre-trained models. Please download these models from [Google Driver](https://drive.google.com/open?id=1ANmiYEGX-IQZTfH8w0aSpj-Wypg-0DR-) (or train by yourself) and save into `.latent-data`.

### Usefull tools
1. Compute the number of parameters and FLOPs of a model:
```
from utils import get_model_infos
flop, param  = get_model_infos(net, (1,3,32,32))
```

2. Different NAS-searched architectures are defined [here](https://github.com/D-X-Y/NAS-Projects/blob/master/lib/nas_infer_model/DXYs/genotypes.py).


## [NAS-Bench-102: Extending the Scope of Reproducible Neural Architecture Search](https://openreview.net/forum?id=HJxyZkBKDr)

We build a new benchmark for neural architecture search, please see more details in [NAS-Bench-102.md](https://github.com/D-X-Y/NAS-Projects/blob/master/NAS-Bench-102.md).

The benchmark data file (v1.0) is `NAS-Bench-102-v1_0-e61699.pth`, which can be downloaded from [Google Drive](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs).

Now you can simply use our API by `pip install nas-bench-102`.

## [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/network-pruning-via-transformable/network-pruning-on-cifar-100)](https://paperswithcode.com/sota/network-pruning-on-cifar-100?p=network-pruning-via-transformable)

In this paper, we proposed a differentiable searching strategy for transformable architectures, i.e., searching for the depth and width of a deep neural network.
You could see the highlight of our Transformable Architecture Search (TAS) at our [project page](https://xuanyidong.com/assets/projects/NeurIPS-2019-TAS.html).

<p float="left">
<img src="https://d-x-y.github.com/resources/paper-icon/NIPS-2019-TAS.png" width="680px"/>
<img src="https://d-x-y.github.com/resources/videos/NeurIPS-2019-TAS/TAS-arch.gif?raw=true" width="180px"/>
</p>


### Usage

Use `bash ./scripts/prepare.sh` to prepare data splits for `CIFAR-10`, `CIFARR-100`, and `ILSVRC2012`.
If you do not have `ILSVRC2012` data, pleasee comment L12 in `./scripts/prepare.sh`.

args: `cifar10` indicates the dataset name, `ResNet56` indicates the basemodel name, `CIFARX` indicates the searching hyper-parameters, `0.47/0.57` indicates the expected FLOP ratio, `-1` indicates the random seed.

#### Search for the depth configuration of ResNet:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-depth-gumbel.sh cifar10 ResNet110 CIFARX 0.57 -1
```

#### Search for the width configuration of ResNet:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-width-gumbel.sh cifar10 ResNet110 CIFARX 0.57 -1
```

#### Search for both depth and width configuration of ResNet:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-shape-cifar.sh cifar10 ResNet56  CIFARX 0.47 -1
```

#### Training the searched shape config from TAS
If you want to directly train a model with searched configuration of TAS, try these:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/tas-infer-train.sh cifar10  C010-ResNet32 -1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/tas-infer-train.sh cifar100 C100-ResNet32 -1
```

### Model Configuration
The searched shapes for ResNet-20/32/56/110/164 in Table 3 in the original paper are listed in [`configs/NeurIPS-2019`](https://github.com/D-X-Y/NAS-Projects/tree/master/configs/NeurIPS-2019).


## [One-Shot Neural Architecture Search via Self-Evaluated Template Network](https://arxiv.org/abs/1910.05733)

<img align="right" src="https://d-x-y.github.com/resources/paper-icon/ICCV-2019-SETN.png" width="450">

<strong>Highlight</strong>: we equip one-shot NAS with an architecture sampler and train network weights using uniformly sampling.


### Usage

Please use the following scripts to train the searched SETN-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  SETN 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 SETN 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k SETN  256 -1
```

The searching codes of SETN on a small search space:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/SETN.sh cifar10 -1
```


## [Searching for A Robust Neural Architecture in Four GPU Hours](https://arxiv.org/abs/1910.04465)


<img align="right" src="https://d-x-y.github.com/resources/paper-icon/CVPR-2019-GDAS.png" width="300">

We proposed a Gradient-based searching algorithm using Differentiable Architecture Sampling (GDAS). GDAS is baseed on DARTS and improves it with Gumbel-softmax sampling.
Experiments on CIFAR-10, CIFAR-100, ImageNet, PTB, and WT2 are reported.


### Usage

#### Reproducing the results of our searched architecture in GDAS
Please use the following scripts to train the searched GDAS-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k GDAS_V1 256 -1
```

#### Searching on the NASNet search space
Please use the following scripts to use GDAS to search as in the original paper:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/GDAS-search-NASNet-space.sh cifar10 1 -1
```

#### Searching on a small search space (NAS-Bench-102)
The GDAS searching codes on a small search space:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh cifar10 -1
```

The baseline searching codes are DARTS:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V1.sh cifar10 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V2.sh cifar10 -1
```

#### Training the searched architecture
To train the searched architecture found by the above scripts, please use the following codes:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NAS-Bench-102/train-a-net.sh '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|' 16 5
```
`|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|` represents the structure of a searched architecture. My codes will automatically print it during the searching procedure.


# Citation

If you find that this project helps your research, please consider citing some of the following papers:
```
@inproceedings{dong2020nasbench102,
  title     = {NAS-Bench-102: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
@inproceedings{dong2019tas,
  title     = {Network Pruning via Transformable Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
@inproceedings{dong2019one,
  title     = {One-Shot Neural Architecture Search via Self-Evaluated Template Network},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {3681--3690},
  year      = {2019}
}
@inproceedings{dong2019search,
  title     = {Searching for A Robust Neural Architecture in Four GPU Hours},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1761--1770},
  year      = {2019}
}
```
