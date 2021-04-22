#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_synthetic_utils.py -s           #
#####################################################
import sys, random
import unittest
import pytest
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from datasets import QuadraticFunc
from datasets import ConstantGenerator, SinGenerator
from datasets import DynamicQuadraticFunc


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
        function.fit([[0, 1], [0.5, 4], [1, 1]], max_iter=3000, verbose=False)
        print(function)
        thresh = 0.15
        self.assertTrue(abs(function(0) - 1) < thresh)
        self.assertTrue(abs(function(0.5) - 4) < thresh)
        self.assertTrue(abs(function(1) - 1) < thresh)


class TestConstantGenerator(unittest.TestCase):
    """Test the constant data generator."""

    def test_simple(self):
        dataset = ConstantGenerator()
        for i, (idx, t, x) in enumerate(dataset):
            assert i == idx, "First loop: {:} vs {:}".format(i, idx)
            assert x == 0.1


class TestSinGenerator(unittest.TestCase):
    """Test the synethtic data generator."""

    def test_simple(self):
        dataset = SinGenerator()
        for i, (idx, t, x) in enumerate(dataset):
            assert i == idx, "First loop: {:} vs {:}".format(i, idx)
        for i, (idx, t, x) in enumerate(dataset):
            assert i == idx, "Second loop: {:} vs {:}".format(i, idx)


class TestDynamicFunc(unittest.TestCase):
    """Test DynamicQuadraticFunc."""

    def test_simple(self):
        timestamps = 30
        function = DynamicQuadraticFunc()
        function_param = dict()
        function_param[0] = SinGenerator(
            num=timestamps, num_sin_phase=4, phase_shift=1.0, max_amplitude=1.0
        )
        function_param[1] = ConstantGenerator(constant=0.9)
        function_param[2] = SinGenerator(
            num=timestamps, num_sin_phase=5, phase_shift=0.4, max_amplitude=0.9
        )
        function.set(function_param)
        print(function)

        with self.assertRaises(TypeError) as context:
            function(0)

        function.set_timestamp(1)
        print(function(2))
