# Nueral Architecture Search

This project contains the following neural architecture search algorithms, implemented in [PyTorch](http://pytorch.org).

- Network Pruning via Transformable Architecture Search, NeurIPS 2019
- One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019
- Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019


## Requirements and Preparation

Please install `PyTorch>=1.0.1`, `Python>=3.6`, and `opencv`.

The CIFAR and ImageNet should be downloaded and extracted into `$TORCH_HOME`.
Some methods use knowledge distillation (KD), which require pre-trained models. Please download these models from [Google Driver](https://drive.google.com/open?id=1ANmiYEGX-IQZTfH8w0aSpj-Wypg-0DR-) (or train by yourself) and save into `.latent-data`.


## [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717)
In this paper, we proposed a differentiable searching strategy for transformable architectures, i.e., searching for the depth and width of a deep neural network.

<img src="https://d-x-y.github.com/resources/paper-icon/NIPS-2019-TAS.png" width="700">

### Usage

Use `bash ./scripts/prepare.sh` to prepare data splits for `CIFAR-10`, `CIFARR-100`, and `ILSVRC2012`.
If you do not have `ILSVRC2012` data, pleasee comment L12 in `./scripts/prepare.sh`.

Search the depth configuration of ResNet:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-depth-gumbel.sh cifar10 ResNet110 CIFARX 0.57 -1
```

Search the width configuration of ResNet:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-width-gumbel.sh cifar10 ResNet110 CIFARX 0.57 -1
```

Search for both depth and width configuration of ResNet:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-cifar.sh cifar10 ResNet56  CIFARX 0.47 -1
```

args: `cifar10` indicates the dataset name, `ResNet56` indicates the basemodel name, `CIFARX` indicates the searching hyper-parameters, `0.47/0.57` indicates the expected FLOP ratio, `-1` indicates the random seed.


## One-Shot Neural Architecture Search via Self-Evaluated Template Network

Highlight: we equip one-shot NAS with an architecture sampler and train network weights using uniformly sampling.

<img src="https://d-x-y.github.com/resources/paper-icon/ICCV-2019-SETN.png" width="450">

### Usage
Please use the following scripts to train the searched SETN-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  SETN 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 SETN 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k SETN  256 -1
```

Searching codes come soon!


## [Searching for A Robust Neural Architecture in Four GPU Hours](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Searching_for_a_Robust_Neural_Architecture_in_Four_GPU_Hours_CVPR_2019_paper.pdf)

We proposed a gradient-based searching algorithm using differentiable architecture sampling (improving DARTS with Gumbel-softmax sampling).

<img src="https://d-x-y.github.com/resources/paper-icon/CVPR-2019-GDAS.png" width="300">

The old version is located at [`others/GDAS`](https://github.com/D-X-Y/NAS-Projects/tree/master/others/GDAS) and a paddlepaddle implementation is locate at [`others/paddlepaddle`](https://github.com/D-X-Y/NAS-Projects/tree/master/others/paddlepaddle).

### Usage

Please use the following scripts to train the searched GDAS-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k GDAS_V1 256 -1
```

Searching codes come soon!


# Citation
If you find that this project helps your research, please consider citing some of the following papers:
```
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
