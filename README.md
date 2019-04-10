## [Searching for A Robust Neural Architecture in Four GPU Hours](http://xuanyidong.com/publication/gradient-based-diff-sampler/)

We propose A Gradient-based neural architecture search approach using Differentiable Architecture Sampler (GDAS).

<img src="data/GDAS.png" width="520">
Figure-1. We utilize a DAG to represent the search space of a neural cell. Different operations (colored arrows) transform one node (square) to its intermediate features (little circles). Meanwhile, each node is the sum of the intermediate features transformed from the previous nodes. As indicated by the solid connections, the neural cell in the proposed GDAS is a sampled sub-graph of this DAG. Specifically, among the intermediate features between every two nodes, GDAS samples one feature in a differentiable way.

### Requirements
- PyTorch 1.0.1
- Python 3.6
- opencv
```
conda install pytorch torchvision cuda100 -c pytorch
```

### Usages

Train the searched CNN on CIFAR
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-cifar.sh GDAS_FG cifar10  cut
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-cifar.sh GDAS_F1 cifar10  cut
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-cifar.sh GDAS_V1 cifar100 cut
```

Train the searched CNN on ImageNet
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts-cnn/train-imagenet.sh GDAS_F1 52 14 B128 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts-cnn/train-imagenet.sh GDAS_V1 50 14 B256 -1
```

Evaluate a trained CNN model
```
CUDA_VISIBLE_DEVICES=0 python ./exps-cnn/evaluate.py --data_path  $TORCH_HOME/cifar.python --checkpoint ${checkpoint-path}
CUDA_VISIBLE_DEVICES=0 python ./exps-cnn/evaluate.py --data_path  $TORCH_HOME/ILSVRC2012 --checkpoint ${checkpoint-path}
CUDA_VISIBLE_DEVICES=0 python ./exps-cnn/evaluate.py --data_path  $TORCH_HOME/ILSVRC2012 --checkpoint GDAS-V1-C50-N14-ImageNet.pth
```

Train the searched RNN
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-rnn/train-PTB.sh DARTS_V1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-rnn/train-PTB.sh DARTS_V2
CUDA_VISIBLE_DEVICES=0 bash ./scripts-rnn/train-PTB.sh GDAS
CUDA_VISIBLE_DEVICES=0 bash ./scripts-rnn/train-WT2.sh DARTS_V1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-rnn/train-WT2.sh DARTS_V2
CUDA_VISIBLE_DEVICES=0 bash ./scripts-rnn/train-WT2.sh GDAS
```

### Training Logs
Some training logs can be found in `./data/logs/`, and some pre-trained models can be found in [Google Driver](https://drive.google.com/open?id=1Ofhc49xC1PLIX4O708gJZ1ugzz4td_RJ).

### Experimental Results
<img src="data/imagenet-results.png" width="700">
Figure 2. Top-1 and top-5 errors on ImageNet.

### Citation
If you find that this project (GDAS) helps your research, please cite the paper:
```
@inproceedings{dong2019search,
  title={Searching for A Robust Neural Architecture in Four GPU Hours},
  author={Dong, Xuanyi and Yang, Yi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
