# GDAS
By Xuanyi Dong and Yi Yang

University of Technology Sydney

Requirements
- PyTorch 1.0
- Python 3.6
- opencv
```
conda install pytorch torchvision cuda100 -c pytorch
```

## Algorithm

Train the searched CNN on CIFAR
```
bash ./scripts-cnn/train-cifar.sh 0 GDAS_FG cifar10  cut
bash ./scripts-cnn/train-cifar.sh 0 GDAS_F1 cifar10  cut
bash ./scripts-cnn/train-cifar.sh 0 GDAS_V1 cifar100 cut
```

Train the searched CNN on ImageNet
```
bash ./scripts-cnn/train-imagenet.sh 0 GDAS_F1 52 14
bash ./scripts-cnn/train-imagenet.sh 0 GDAS_V1 50 14
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
