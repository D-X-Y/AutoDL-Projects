#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_super_vit.py -s               #
#####################################################
import unittest
from parameterized import parameterized

import torch
from xautodl.xmodels import transformers
from xautodl.utils.flop_benchmark import count_parameters


class TestSuperViT(unittest.TestCase):
    """Test the super re-arrange layer."""

    def test_super_vit(self):
        model = transformers.get_transformer("vit-base-16")
        tensor = torch.rand((2, 3, 224, 224))
        print("The tensor shape: {:}".format(tensor.shape))
        # print(model)
        outs = model(tensor)
        print("The output tensor shape: {:}".format(outs.shape))

    @parameterized.expand(
        [
            ["vit-cifar10-p4-d4-h4-c32", 32],
            ["vit-base-16", 224],
            ["vit-large-16", 224],
            ["vit-huge-14", 224],
        ]
    )
    def test_imagenet(self, name, resolution):
        tensor = torch.rand((2, 3, resolution, resolution))
        config = transformers.name2config[name]
        model = transformers.get_transformer(config)
        outs = model(tensor)
        size = count_parameters(model, "mb", True)
        print(
            "{:10s} : size={:.2f}MB, out-shape: {:}".format(
                name, size, tuple(outs.shape)
            )
        )
