# Searching for A Robust Neural Architecture in Four GPU Hours

We propose A Gradient-based neural architecture search approach using Differentiable Architecture Sampler (GDAS).

## Requirements
- PyTorch 1.0.1
- Python 3.6
- opencv
```
conda install pytorch torchvision cuda100 -c pytorch
```

## Usages

Train the searched CNN on CIFAR
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-cifar.sh GDAS_FG cifar10  cut
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-cifar.sh GDAS_F1 cifar10  cut
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-cifar.sh GDAS_V1 cifar100 cut
```

Train the searched CNN on ImageNet
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-imagenet.sh GDAS_F1 52 14
CUDA_VISIBLE_DEVICES=0 bash ./scripts-cnn/train-imagenet.sh GDAS_V1 50 14
```

Evaluate a trained CNN model
```
CUDA_VISIBLE_DEVICES=0 python ./exps-cnn/evaluate.py --data_path  $TORCH_HOME/cifar.python --checkpoint ${checkpoint-path}
CUDA_VISIBLE_DEVICES=0 python ./exps-cnn/evaluate.py --data_path  $TORCH_HOME/ILSVRC2012 --checkpoint ${checkpoint-path}
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

## Training Logs
Some training logs can be found in `./data/logs/`, and some pre-trained models can be found in [Google Driver](https://drive.google.com/open?id=1Ofhc49xC1PLIX4O708gJZ1ugzz4td_RJ).

## Citation
```
@inproceedings{dong2019search,
  title={Searching for A Robust Neural Architecture in Four GPU Hours},
  author={Dong, Xuanyi and Yang, Yi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
