# [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/network-pruning-via-transformable/network-pruning-on-cifar-100)](https://paperswithcode.com/sota/network-pruning-on-cifar-100?p=network-pruning-via-transformable)

Network Pruning via Transformable Architecture Search is accepted by NeurIPS 2019.
In this paper, we proposed a differentiable searching strategy for transformable architectures, i.e., searching for the depth and width of a deep neural network.
You could see the highlight of our Transformable Architecture Search (TAS) at our [project page](https://xuanyidong.com/assets/projects/NeurIPS-2019-TAS.html).

<p float="left">
<img src="https://d-x-y.github.com/resources/paper-icon/NIPS-2019-TAS.png" width="680px"/>
<img src="https://d-x-y.github.com/resources/videos/NeurIPS-2019-TAS/TAS-arch.gif?raw=true" width="180px"/>
</p>


## Requirements and Preparation

Please install `Python>=3.6` and `PyTorch>=1.2.0`.

CIFAR and ImageNet should be downloaded and extracted into `$TORCH_HOME`.
The proposed method utilized knowledge distillation (KD), which require pre-trained models. Please download these models from [Google Drive](https://drive.google.com/open?id=1ANmiYEGX-IQZTfH8w0aSpj-Wypg-0DR-) (or train by yourself) and save into `.latent-data`.

**LOGS**:
We provide some logs at [Google Drive](https://drive.google.com/open?id=1_qUY4DTtuW_l6ZonynQAC9ttqy35fxZ-). It includes (1) logs of training searched shape of ResNet-18 and ResNet-50 on ImageNet, (2) logs of searching and training for ResNet-164 on CIFAR, (3) logs of searching and training for ResNet56 on CIFAR-10, (4) logs of searching and training for ResNet110 on CIFAR-100.

## Usage

Use `bash ./scripts/prepare.sh` to prepare data splits for `CIFAR-10`, `CIFARR-100`, and `ILSVRC2012`.
If you do not have `ILSVRC2012` data, please comment L12 in `./scripts/prepare.sh`.

args: `cifar10` indicates the dataset name, `ResNet56` indicates the basemodel name, `CIFARX` indicates the searching hyper-parameters, `0.47/0.57` indicates the expected FLOP ratio, `-1` indicates the random seed.

**Model Configuration**

The searched shapes for ResNet-20/32/56/110/164 and ResNet-18/50 in Table 3/4 in the original paper are listed in [`configs/NeurIPS-2019`](https://github.com/D-X-Y/AutoDL-Projects/tree/master/configs/NeurIPS-2019).

**Search for the depth configuration of ResNet**
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-depth-gumbel.sh cifar10 ResNet110 CIFARX 0.57 -1
```

**Search for the width configuration of ResNet**
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-width-gumbel.sh cifar10 ResNet110 CIFARX 0.57 -1
```

**Search for both depth and width configuration of ResNet**
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts-search/search-shape-cifar.sh cifar10 ResNet56  CIFARX 0.47 -1
```

**Training the searched shape config from TAS:**
If you want to directly train a model with searched configuration of TAS, try these:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/tas-infer-train.sh cifar10  C010-ResNet32 -1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/tas-infer-train.sh cifar100 C100-ResNet32 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/tas-infer-train.sh imagenet-1k ImageNet-ResNet18V1 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/tas-infer-train.sh imagenet-1k ImageNet-ResNet50V1 -1
```


# Citation

If you find that this project helps your research, please consider citing the following paper:
```
@inproceedings{dong2019tas,
  title     = {Network Pruning via Transformable Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
```
