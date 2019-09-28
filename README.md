# Nueral Architecture Search

This project contains the following neural architecture search algorithms, implemented in PyTorch.

- Network Pruning via Transformable Architecture Search
- One-Shot Neural Architecture Search via Self-Evaluated Template Network
- Searching for A Robust Neural Architecture in Four GPU Hours


## Requirements and Preparation

Please install `PyTorch>=1.0.1`, `Python>=3.6`, and `opencv`.

The CIFAR and ImageNet should be downloaded and extracted into `$TORCH_HOME`.
Some methods use knowledge distillation (KD), which require pre-trained models. Please download these models from [Google Driver](https://drive.google.com/open?id=1ANmiYEGX-IQZTfH8w0aSpj-Wypg-0DR-) (or train by yourself) and save into `.latent-data`.


## Network Pruning via Transformable Architecture Search

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

Train the searched SETN-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  SETN 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 SETN 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k SETN  256 -1
```

Searching codes come soon!


## Searching for A Robust Neural Architecture in Four GPU Hours

The old version is located in `others/GDAS`.

Train the searched GDAS-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
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
  title={Searching for A Robust Neural Architecture in Four GPU Hours},
  author={Dong, Xuanyi and Yang, Yi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1761--1770},
  year={2019}
}
```
