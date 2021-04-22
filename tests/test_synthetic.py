#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_synthetic.py -s                 #
#####################################################
import sys, random
import unittest
import pytest
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from datasets import QuadraticFunction
from datasets import SynAdaptiveEnv


class TestQuadraticFunction(unittest.TestCase):
    """Test the quadratic function."""

    def test_simple(self):
        function = QuadraticFunction([[0, 1], [0.5, 4], [1, 1]])
        print(function)
        for x in (0, 0.5, 1):
            print("f({:})={:}".format(x, function[x]))
        thresh = 0.2
        self.assertTrue(abs(function[0] - 1) < thresh)
        self.assertTrue(abs(function[0.5] - 4) < thresh)
        self.assertTrue(abs(function[1] - 1) < thresh)

    def test_none(self):
        function = QuadraticFunction()
        function.fit([[0, 1], [0.5, 4], [1, 1]], max_iter=3000, verbose=True)
        print(function)
        thresh = 0.2
        self.assertTrue(abs(function[0] - 1) < thresh)
        self.assertTrue(abs(function[0.5] - 4) < thresh)
        self.assertTrue(abs(function[1] - 1) < thresh)


class TestSynAdaptiveEnv(unittest.TestCase):
    """Test the synethtic adaptive environment."""

    def test_simple(self):
        dataset = SynAdaptiveEnv()
        for i, (idx, t, x) in enumerate(dataset):
            assert i == idx, "First loop: {:} vs {:}".format(i, idx)
        for i, (idx, t, x) in enumerate(dataset):
            assert i == idx, "Second loop: {:} vs {:}".format(i, idx)
