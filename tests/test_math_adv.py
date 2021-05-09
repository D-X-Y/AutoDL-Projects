#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_math_adv.py -s                  #
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
from datasets.math_core import ConstantFunc
from datasets.math_core import DynamicLinearFunc
from datasets.math_core import DynamicQuadraticFunc
from datasets.math_core import ComposedSinFunc


class TestConstantFunc(unittest.TestCase):
    """Test the constant function."""

    def test_simple(self):
        function = ConstantFunc(0.1)
        for i in range(100):
            assert function(i) == 0.1


class TestDynamicFunc(unittest.TestCase):
    """Test DynamicQuadraticFunc."""

    def test_simple(self):
        timestamps = 30
        function = DynamicQuadraticFunc()
        function_param = dict()
        function_param[0] = ComposedSinFunc(
            num=timestamps, num_sin_phase=4, phase_shift=1.0, max_amplitude=1.0
        )
        function_param[1] = ConstantFunc(constant=0.9)
        function_param[2] = ComposedSinFunc(
            num=timestamps, num_sin_phase=5, phase_shift=0.4, max_amplitude=0.9
        )
        function.set(function_param)
        print(function)

        with self.assertRaises(TypeError) as context:
            function(0)

        function.set_timestamp(1)
        print(function(2))

    def test_simple_linear(self):
        timestamps = 30
        function = DynamicLinearFunc()
        function_param = dict()
        function_param[0] = ComposedSinFunc(
            num=timestamps, num_sin_phase=4, phase_shift=1.0, max_amplitude=1.0
        )
        function_param[1] = ConstantFunc(constant=0.9)
        function.set(function_param)
        print(function)

        with self.assertRaises(TypeError) as context:
            function(0)

        function.set_timestamp(1)
        print(function(2))
