#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_super_rearrange.py -s         #
#####################################################
import sys
import unittest
from pathlib import Path

lib_dir = (Path(__file__).parent / "..").resolve()
print("LIB-DIR: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import torch
from xautodl import xlayers


class TestSuperReArrange(unittest.TestCase):
    """Test the super re-arrange layer."""

    def test_super_re_arrange(self):
        layer = xlayers.SuperReArrange(
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=4, p2=4
        )
        tensor = torch.rand((8, 4, 32, 32))
        print("The tensor shape: {:}".format(tensor.shape))
        print(layer)
