#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################
# python exps/prepare.py --name cifar10     --root $TORCH_HOME/cifar.python --save ./data/cifar10.split.pth
# python exps/prepare.py --name cifar100    --root $TORCH_HOME/cifar.python --save ./data/cifar100.split.pth
# python exps/prepare.py --name imagenet-1k --root $TORCH_HOME/ILSVRC2012   --save ./data/imagenet-1k.split.pth
#####################################################
import sys, time, torch, random, argparse
from collections import defaultdict
import os.path as osp
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import torchvision
import torchvision.datasets as dset

parser = argparse.ArgumentParser(
    description="Prepare splits for searching",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--name", type=str, help="The dataset name.")
parser.add_argument("--root", type=str, help="The directory to the dataset.")
parser.add_argument("--save", type=str, help="The save path.")
parser.add_argument("--ratio", type=float, help="The save path.")
args = parser.parse_args()


def main():
    save_path = Path(args.save)
    save_dir = save_path.parent
    name = args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    assert not save_path.exists(), "{:} already exists".format(save_path)
    print("torchvision version : {:}".format(torchvision.__version__))

    if name == "cifar10":
        dataset = dset.CIFAR10(args.root, train=True, download=True)
    elif name == "cifar100":
        dataset = dset.CIFAR100(args.root, train=True, download=True)
    elif name == "imagenet-1k":
        dataset = dset.ImageFolder(osp.join(args.root, "train"))
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "train_labels"):
        targets = dataset.train_labels
    elif hasattr(dataset, "imgs"):
        targets = [x[1] for x in dataset.imgs]
    else:
        raise ValueError("invalid pattern")
    print("There are {:} samples in this dataset.".format(len(targets)))

    class2index = defaultdict(list)
    train, valid = [], []
    random.seed(111)
    for index, cls in enumerate(targets):
        class2index[cls].append(index)
    classes = sorted(list(class2index.keys()))
    for cls in classes:
        xlist = class2index[cls]
        xtrain = random.sample(xlist, int(len(xlist) * args.ratio))
        xvalid = list(set(xlist) - set(xtrain))
        train += xtrain
        valid += xvalid
    train.sort()
    valid.sort()
    ## for statistics
    class2numT, class2numV = defaultdict(int), defaultdict(int)
    for index in train:
        class2numT[targets[index]] += 1
    for index in valid:
        class2numV[targets[index]] += 1
    class2numT, class2numV = dict(class2numT), dict(class2numV)
    torch.save(
        {
            "train": train,
            "valid": valid,
            "class2numTrain": class2numT,
            "class2numValid": class2numV,
        },
        save_path,
    )
    print("-" * 80)


if __name__ == "__main__":
    main()
