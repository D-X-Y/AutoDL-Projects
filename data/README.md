# ImageNet

The class names of ImageNet-1K are in `classes.txt`.

# A 100-class subset of ImageNet-1K : ImageNet-100

The class names of ImageNet-100 are in `ImageNet-100.txt`.

Run `python split-imagenet.py` will automatically create ImageNet-100 based on the data of ImageNet-1K. By default, we assume the data of ImageNet-1K locates at `~/.torch/ILSVRC2012`. If your data is in a different location, you need to modify line-19 and line-20 in `split-imagenet.py`.

# Tiny-ImageNet
The official website is [here](https://tiny-imagenet.herokuapp.com/). Please run `python tiny-imagenet.py` to generate the correct format of Tiny ImageNet for training.

# PTB and WT2
Run `bash Get-PTB-WT2.sh` to download the data.
