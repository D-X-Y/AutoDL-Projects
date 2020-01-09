# Basic Classification Models

## Performance on CIFAR

|        Model       | FLOPs       | Params (M) | Error on CIFAR-10 | Error on CIFAR-100 | Batch-GPU |
|:------------------:|:-----------:|:----------:|:-----------------:|:------------------:|:---------:|
| ResNet-08          |  12.50 M    |  0.08      |       12.14       |       40.20        |  256-2    |
| ResNet-20          |  40.81 M    |  0.27      |       7.26        |       31.38        |  256-2    |
| ResNet-32          |  69.12 M    |  0.47      |       6.19        |       29.56        |  256-2    |
| ResNet-56          | 125.75 M    |  0.86      |       5.74        |       26.82        |  256-2    |
| ResNet-110         | 253.15 M    |  1.73      |       5.14        |       25.18        |  256-2    |
| ResNet-110         | 253.15 M    |  1.73      |       5.06        |       25.49        |  256-1    |
| ResNet-164         | 247.65 M    |  1.70      |       4.36        |       21.48        |  256-2    |
| ResNet-1001        | 1491.00 M   |  10.33     |       5.34        |       22.50        |  256-2    |
| DenseNet-BC100-12  | 287.93 M    |  0.77      |       4.68        |       22.76        |  256-2    |
| DenseNet-BC100-12  | 287.93 M    |  0.77      |       4.25        |       21.54        |  128-2    |
| DenseNet-BC100-12  | 287.93 M    |  0.77      |       5.51        |       24.67        |   64-1    |
| WRN-28-10          | 5243.33 M   |  36.48     |       3.61        |       19.65        |  256-2    |

```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/base-train.sh cifar10 ResNet20  E300 L1 256 -1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/base-train.sh cifar10 ResNet56  E300 L1 256 -1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/base-train.sh cifar10 ResNet110 E300 L1 256 -1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/base-train.sh cifar10 ResNet164 E300 L1 256 -1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/base-train.sh cifar10 DenseBC100-12 E300 L1 256 -1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/base-train.sh cifar10 WRN28-10  E300 L1 256 -1
CUDA_VISIBLE_DEVICES=0,1 python ./exps/basic-eval.py --data_path ${TORCH_HOME}/ILSVRC2012 --checkpoint
CUDA_VISIBLE_DEVICES=0,1 python ./exps/test-official-CNN.py --data_path ${TORCH_HOME}/ILSVRC2012
```

Train some NAS models:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  SETN 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 SETN 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k SETN  256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k SETN1 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k DARTS 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k GDAS_V1 256 -1
```

## Performance on ImageNet

|        Model      | FLOPs (GB) | Params (M) | Top-1 Error | Top-5 Error |  Optimizer |
|:-----------------:|:----------:|:----------:|:-----------:|:-----------:|:----------:|
| ResNet-18         | 1.814      |  11.69     |   30.24     |   10.92     | Official   |
| ResNet-18         | 1.814      |  11.69     |   29.97     |   10.43     | Step-120   |
| ResNet-18         | 1.814      |  11.69     |   29.35     |   10.13     | Cosine-120 |
| ResNet-18         | 1.814      |  11.69     |   29.45     |   10.25     | Cosine-120 B1024 |
| ResNet-18         | 1.814      |  11.69     |   29.44     |   10.12     | Cosine-S-120 |
| ResNet-18 (DS)    | 2.053      |  11.71     |   28.53     |   9.69      | Cosine-S-120 |
| ResNet-34         | 3.663      |  21.80     |   25.65     |   8.06      | Cosine-120   |
| ResNet-34 (DS)    | 3.903      |  21.82     |   25.05     |   7.67      | Cosine-S-120 |
| ResNet-50         | 4.089      |  25.56     |   23.85     |   7.13      | Official     |
| ResNet-50         | 4.089      |  25.56     |   22.54     |   6.45      | Cosine-120   |
| ResNet-50         | 4.089      |  25.56     |   22.71     |   6.38      | Cosine-120 B1024 |
| ResNet-50         | 4.089      |  25.56     |   22.34     |   6.22      | Cosine-S-120 |
| ResNet-50 (DS)    | 4.328      |  25.58     |   22.67     |   6.39      | Step-120     |
| ResNet-50 (DS)    | 4.328      |  25.58     |   21.94     |   6.23      | Cosine-120   |
| ResNet-50 (DS)    | 4.328      |  25.58     |   21.71     |   5.99      | Cosine-S-120 |
| ResNet-101        | 7.801      |  44.55     |   20.93     |   5.57      | Cosine-120   |
| ResNet-101        | 7.801      |  44.55     |   20.92     |   5.58      | Cosine-120 B1024 |
| ResNet-101 (DS)   | 8.041      |  44.57     |   20.36     |   5.22      | Cosine-S-120 |
| ResNet-152        | 11.514     |  60.19     |   20.10     |   5.17      | Cosine-120 B1024 |
| ResNet-152 (DS)   | 11.753     |  60.21     |   19.83     |   5.02      | Cosine-S-120 |
| ResNet-200        | 15.007     |  64.67     |   20.06     |   4.98      | Cosine-S-120 |
| Next50-32x4d (DS) | 4.2        |  25.0      |   22.2      |     -       | Official     |
| Next50-32x4d (DS) | 4.470      |  25.05     |   21.16     |   5.65      | Cosine-S-120 |
| MobileNet-V2      | 0.300      |  3.40      |   28.0      |     -       | Official     |
| MobileNet-V2      | 0.300      |  3.50      |   27.92     |   9.50      | MobileFast   |
| MobileNet-V2      | 0.300      |  3.50      |   27.56     |   9.26      | MobileFast-Smooth |
| ShuffleNet-V2 1.0 | 0.146      |  2.28      |   30.6      |   11.1      | Official     |
| ShuffleNet-V2 1.0 | 0.145      |  2.28      |             |             | Cosine-S-120 |
| ShuffleNet-V2 1.5 | 0.299      |            |   27.4      |     -       | Official     |
| ShuffleNet-V2 1.5 |            |            |             |             | Cosine-S-120 |
| ShuffleNet-V2 2.0 |            |            |             |             | Cosine-S-120 |

`DS` indicates deep-stem for the first convolutional layer.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet18V1 Step-Soft 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet18V1  Cos-Soft 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet18V1  Cos-Soft 1024 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet18V1  Cos-Smooth 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet18V2  Cos-Smooth 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet34V2  Cos-Smooth 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet50V1 Cos-Soft 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet50V2 Step-Soft 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet50V2 Cos-Soft 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNet101V2 Cos-Smooth 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh ResNext50-32x4dV2 Cos-Smooth 256 -1
```

Train efficient models may require different hyper-parameters.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh MobileNetV2-X MobileFast 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh MobileNetV2-X MobileFastS 256 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/base-imagenet.sh MobileNetV2   Mobile     256 -1 (70.96 top-1, 90.05 top-5)
```

# Train with Knowledge Distillation

ResNet110 -> ResNet20
```
bash ./scripts-cluster/local.sh 0,1 "bash ./scripts/KD-train.sh cifar10 ResNet20 ResNet110 0.9 4 -1"
```

ResNet110 -> ResNet110
```
bash ./scripts-cluster/local.sh 0,1 "bash ./scripts/KD-train.sh cifar10 ResNet110 ResNet110 0.9 4 -1"
```

Set alpha=0.9 and temperature=4 following `Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR 2017`.

# Linux
The following command will redirect the output of top command to `top.txt`.
```
top -b -n 1 > top.txt
```

## Download the ImageNet dataset
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```
