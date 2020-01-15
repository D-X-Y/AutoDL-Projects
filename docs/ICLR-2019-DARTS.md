# DARTS: Differentiable Architecture Search

DARTS: Differentiable Architecture Search is accepted by ICLR 2019.
In this paper, Hanxiao proposed a differentiable neural architecture search method, named as DARTS.
Recently, DARTS becomes very popular due to its simplicity and performance.

**Run DARTS on the NAS-Bench-201 search space**:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V2.sh cifar10 1 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh     cifar10 1 -1
```

# Citation

```
@inproceedings{liu2019darts,
  title     = {{DARTS}: Differentiable architecture search},
  author    = {Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}
```
