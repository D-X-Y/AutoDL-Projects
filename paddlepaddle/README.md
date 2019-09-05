# Image Classification based on NAS-Searched Models

This directory contains 10 image classification models.
Nine of them are automatically searched models using different Neural Architecture Search (NAS) algorithms, and the other is the residual network.
We provide codes and scripts to train these models on both CIFAR-10 and CIFAR-100.
We use the standard data augmentation, i.e., random crop, random flip, and normalization.

---
## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training Models](#training-models)
- [Project Structure](#project-structure)
- [Citation](#citation)


### Installation
This project has the following requirements:
- Python = 3.6
- PadddlePaddle Fluid >= v0.15.0
- numpy, tarfile, cPickle, PIL


### Data Preparation
Please download [CIFAR-10](https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz) and [CIFAR-100](https://dataset.bj.bcebos.com/cifar/cifar-100-python.tar.gz) before running the codes.
Note that the MD5 of CIFAR-10-Python compressed file is `c58f30108f718f92721af3b95e74349a` and the MD5 of CIFAR-100-Python compressed file is `eb9058c3a382ffc7106e4002c42a8d85`.
Please save the file into `${TORCH_HOME}/cifar.python`.
After data preparation, there should be two files `${TORCH_HOME}/cifar.python/cifar-10-python.tar.gz` and `${TORCH_HOME}/cifar.python/cifar-100-python.tar.gz`.


### Training Models

After setting up the environment and preparing the data, you can train the model. The main function entrance is `train_cifar.py`. We also provide some scripts for easy usage.
```
bash ./scripts/base-train.sh 0 cifar-10 ResNet110
bash ./scripts/train-nas.sh  0 cifar-10 GDAS_V1
bash ./scripts/train-nas.sh  0 cifar-10 GDAS_V2
bash ./scripts/train-nas.sh  0 cifar-10  SETN
bash ./scripts/train-nas.sh  0 cifar-10 NASNet
bash ./scripts/train-nas.sh  0 cifar-10 ENASNet
bash ./scripts/train-nas.sh  0 cifar-10 AmoebaNet
bash ./scripts/train-nas.sh  0 cifar-10 PNASNet
bash ./scripts/train-nas.sh  0 cifar-100 SETN
```
The first argument is the GPU-ID to train your program, the second argument is the dataset name (`cifar-10` or `cifar-100`), and the last one is the model name.
Please use `./scripts/base-train.sh` for ResNet and use `./scripts/train-nas.sh` for NAS-searched models.


### Project Structure
```
.
├──train_cifar.py [Training CNN models]
├──lib [Library for dataset, models, and others]
│  └──models  
│     ├──__init__.py [Import useful Classes and Functions in models]  
│     ├──resnet.py [Define the ResNet models]
│     ├──operations.py [Define the atomic operation in NAS search space]
│     ├──genotypes.py [Define the topological structure of different NAS-searched models]
│     └──nas_net.py [Define the macro structure of NAS models]
│  └──utils
│     ├──__init__.py [Import useful Classes and Functions in utils]  
│     ├──meter.py [Define the AverageMeter class to count the accuracy and loss]
│     ├──time_utils.py [Define some functions to print date or convert seconds into hours]
│     └──data_utils.py [Define data augmentation functions and dataset reader for CIFAR]
└──scripts [Scripts for running]  
```


### Citation
If you find that this project helps your research, please consider citing these papers:
```
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
@inproceedings{liu2018darts,
  title     = {Darts: Differentiable architecture search},
  author    = {Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018}
}
@inproceedings{pham2018efficient,
  title     = {Efficient Neural Architecture Search via Parameter Sharing},
  author    = {Pham, Hieu and Guan, Melody and Zoph, Barret and Le, Quoc and Dean, Jeff},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {4092--4101},
  year      = {2018}
}
@inproceedings{liu2018progressive,
  title     = {Progressive neural architecture search},
  author    = {Liu, Chenxi and Zoph, Barret and Neumann, Maxim and Shlens, Jonathon and Hua, Wei and Li, Li-Jia and Fei-Fei, Li and Yuille, Alan and Huang, Jonathan and Murphy, Kevin},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages     = {19--34},
  year      = {2018}
}
@inproceedings{zoph2018learning,
  title     = {Learning transferable architectures for scalable image recognition},
  author    = {Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {8697--8710},
  year      = {2018}
}
@inproceedings{real2019regularized,
  title     = {Regularized evolution for image classifier architecture search},
  author    = {Real, Esteban and Aggarwal, Alok and Huang, Yanping and Le, Quoc V},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  pages     = {4780--4789},
  year      = {2019}
}
```
