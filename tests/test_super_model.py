#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_super_model.py -s             #
#####################################################
import sys, random
import unittest
import pytest
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import torch
from xlayers import super_core
import spaces


class TestSuperLinear(unittest.TestCase):
    """Test the super linear."""

    def test_super_linear(self):
        out_features = spaces.Categorical(12, 24, 36)
        bias = spaces.Categorical(True, False)
        model = super_core.SuperLinear(10, out_features, bias=bias)
        print(model)
        print(model.super_run_type)
        print(model.abstract_search_space())
