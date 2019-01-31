# Neural-Architecture-Search

### Baseline
```
bash ./scripts-nas/search.sh 1 base cifar10
bash ./scripts-nas/search.sh 1 share
bash ./scripts-nas/batch-base-search.sh 1
bash ./scripts-nas/batch-base-model.sh 1
```

### Meta
```
bash ./scripts-nas/meta-search.sh 0 meta 20 5
```

### Acceleration
```
bash ./scripts-nas/search-acc-v2.sh 3 acc2
bash ./scripts-nas/DMS-V-Train.sh 0

bash ./scripts-nas/search-acc-simple.sh 3 NetworkV2
```

### Base Model Training
```
bash ./scripts-nas/train-model.sh 3 AmoebaNet
bash ./scripts-nas/train-model.sh 3 NASNet
bash ./scripts-nas/train-model.sh 3 DARTS_V1
bash ./scripts-nas/train-model-simple.sh 3 AmoebaNet
bash ./scripts-nas/train-imagenet.sh 3 DARTS_V2 50 14

bash scripts-nas/TRAIN-BASE.sh 0 PNASNet cifar10 nocut 48 11
bash scripts-nas/TRAIN-BASE.sh 0 AmoebaNet cifar10 nocut 36 20
bash scripts-nas/TRAIN-BASE.sh 0 NASNet cifar10 nocut 33 20

bash scripts-nas/TRAIN-BASE.sh 0 DMS_F1 cifar10 nocut 36 20
bash scripts-nas/TRAIN-BASE.sh 0 DMS_V1 cifar10 nocut 36 20
bash scripts-nas/TRAIN-BASE.sh 0 GDAS_CC cifar10 nocut 36 20
bash scripts-nas/train-imagenet.sh 3 DMS_F1 52 14
bash scripts-nas/train-imagenet.sh 3 DMS_V1 50 14


bash scripts-nas/TRAIN-BASE.sh 0 DMS_V1 cifar10 nocut 36 20
```


### Visualization
```
python ./exps-nas/vis-arch.py --checkpoint  --save_dir
python ./exps-nas/cvpr-vis.py --save_dir ./snapshots/NAS-VIS/
```

### Test datasets
```
cd ./lib/datasets/
python test_NLP.py
```
