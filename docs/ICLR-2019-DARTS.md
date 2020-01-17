# DARTS: Differentiable Architecture Search

DARTS: Differentiable Architecture Search is accepted by ICLR 2019.
In this paper, Hanxiao proposed a differentiable neural architecture search method, named as DARTS.
Recently, DARTS becomes very popular due to its simplicity and performance.

## Run DARTS on the NAS-Bench-201 search space
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V2.sh cifar10 1 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh     cifar10 1 -1
```

## Run the first-order DARTS on the NASNet/DARTS search space
This command will start to use the first-order DARTS to search architectures on the DARTS search space.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/DARTS1V-search-NASNet-space.sh cifar10 -1
```

After searching, if you want to train the searched architecture found by the above scripts, you need to add the config of that architecture (will be printed in log) in [genotypes.py](https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/nas_infer_model/DXYs/genotypes.py).
In future, I will add a more eligent way to train the searched architecture from the DARTS search space.


# Citation

```
@inproceedings{liu2019darts,
  title     = {{DARTS}: Differentiable architecture search},
  author    = {Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}
```
