#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_loader.py -s                    #
#####################################################
import unittest
import tempfile
import torch

from xautodl.datasets import get_datasets


def test_simple():
    xdir = tempfile.mkdtemp()
    train_data, valid_data, xshape, class_num = get_datasets("cifar10", xdir, -1)
    print(train_data)
    print(valid_data)

    xloader = torch.utils.data.DataLoader(
        train_data, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    print(xloader)
    print(next(iter(xloader)))

    for i, data in enumerate(xloader):
        print(i)


test_simple()
