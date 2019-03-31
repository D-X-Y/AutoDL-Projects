# Commands on Cluster

## RNN
```
bash scripts-cluster/submit.sh yq01-v100-box-idl-2-8 WT2-GDAS 1 "bash ./scripts-rnn/train-WT2.sh GDAS"
bash scripts-cluster/submit.sh yq01-v100-box-idl-2-8 PTB-GDAS 1 "bash ./scripts-rnn/train-PTB.sh GDAS"
```

## CNN
```
bash scripts-cluster/submit.sh yq01-v100-box-idl-2-8 CIFAR10-CUT-GDAS-F1 1 "bash ./scripts-cnn/train-cifar.sh GDAS_F1 cifar10  cut"
bash scripts-cluster/submit.sh yq01-v100-box-idl-2-8 IMAGENET-GDAS-F1    1 "bash ./scripts-cnn/train-imagenet.sh GDAS_F1 52 14"
```
