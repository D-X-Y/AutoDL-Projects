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
from xlayers.super_core import SuperRunMode
from trade_models import get_transformer


class TestSuperTransformer(unittest.TestCase):
    """Test the super transformer."""

    def test_super_transformer(self):
        model = get_transformer(None)
        model.apply_verbose(False)
        print(model)

        inputs = torch.rand(10, 360)
        print("Input shape: {:}".format(inputs.shape))
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (10,))

        abstract_space = model.abstract_search_space
        abstract_space.clean_last()
        abstract_child = abstract_space.random(reuse_last=True)
        print("The abstract searc space:\n{:}".format(abstract_space))
        print("The abstract child program:\n{:}".format(abstract_child))

        model.set_super_run_type(SuperRunMode.Candidate)
        model.apply_candidate(abstract_child)

        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (10,))
