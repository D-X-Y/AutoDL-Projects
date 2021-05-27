#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_math_static.py -s               #
#####################################################
import unittest

from xautodl.datasets.math_core import QuadraticSFunc
from xautodl.datasets.math_core import ConstantFunc


class TestConstantFunc(unittest.TestCase):
    """Test the constant function."""

    def test_simple(self):
        function = ConstantFunc(0.1)
        for i in range(100):
            assert function(i) == 0.1


class TestQuadraticSFunc(unittest.TestCase):
    """Test the quadratic function."""

    def test_simple(self):
        function = QuadraticSFunc({0: 1, 1: 2, 2: 1})
        print(function)
        for x in (0, 0.5, 1):
            print("f({:})={:}".format(x, function(x)))
        thresh = 1e-7
        self.assertTrue(abs(function(0) - 1) < thresh)
        self.assertTrue(abs(function(0.5) - 0.5 * 0.5 - 2 * 0.5 - 1) < thresh)
        self.assertTrue(abs(function(1) - 1 - 2 - 1) < thresh)
