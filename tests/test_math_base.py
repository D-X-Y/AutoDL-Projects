#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_math_base.py -s                 #
#####################################################
import sys, random
import unittest
import pytest
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from datasets.math_core import QuadraticFunc


class TestQuadraticFunc(unittest.TestCase):
    """Test the quadratic function."""

    def test_simple(self):
        function = QuadraticFunc([[0, 1], [0.5, 4], [1, 1]])
        print(function)
        for x in (0, 0.5, 1):
            print("f({:})={:}".format(x, function(x)))
        thresh = 0.2
        self.assertTrue(abs(function(0) - 1) < thresh)
        self.assertTrue(abs(function(0.5) - 4) < thresh)
        self.assertTrue(abs(function(1) - 1) < thresh)

    def test_none(self):
        function = QuadraticFunc()
        function.fit(
            list_of_points=[[0, 1], [0.5, 4], [1, 1]], max_iter=3000, verbose=False
        )
        print(function)
        thresh = 0.15
        self.assertTrue(abs(function(0) - 1) < thresh)
        self.assertTrue(abs(function(0.5) - 4) < thresh)
        self.assertTrue(abs(function(1) - 1) < thresh)
