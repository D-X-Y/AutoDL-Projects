#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_super_vit.py -s               #
#####################################################
import sys
import unittest

import torch
from xautodl.xmodels import transformers
from xautodl.utils.flop_benchmark import count_parameters

class TestSuperViT(unittest.TestCase):
    """Test the super re-arrange layer."""

    def test_super_vit(self):
        model = transformers.get_transformer("vit-base")
        tensor = torch.rand((16, 3, 256, 256))
        print("The tensor shape: {:}".format(tensor.shape))
        print(model)
        outs = model(tensor)
        print("The output tensor shape: {:}".format(outs.shape))

    def test_model_size(self):
        name2config = transformers.name2config
        for name, config in name2config.items():
            model = transformers.get_transformer(config)
            size = count_parameters(model, "mb", True)
            print('{:10s} : size={:.2f}MB'.format(name, size))
