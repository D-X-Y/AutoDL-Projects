# GDAS
By Xuanyi Dong and Yi Yang

University of Technology Sydney

Requirements
- PyTorch 1.0
- Python 3.6
```
conda install pytorch torchvision cuda100 -c pytorch
```

## Algorithm

Searching CNNs
```
```

Train the searched CNN on CIFAR
```
bash ./scripts-cnn/train-imagenet.sh 0 GDAS_F1 52 14
bash ./scripts-cnn/train-imagenet.sh 0 GDAS_V1 50 14
```

Train the searched CNN on ImageNet
```
bash ./scripts-cnn/train-imagenet.sh 0 GDAS_F1 52 14
bash ./scripts-cnn/train-imagenet.sh 0 GDAS_V1 50 14
```


Train the searched RNN
```
bash ./scripts-rnn/train-PTB.sh 0 DARTS_V1
bash ./scripts-rnn/train-PTB.sh 0 DARTS_V2
bash ./scripts-rnn/train-PTB.sh 0 GDAS
bash ./scripts-rnn/train-WT2.sh 0 DARTS_V1
bash ./scripts-rnn/train-WT2.sh 0 DARTS_V2
bash ./scripts-rnn/train-WT2.sh 0 GDAS
```
