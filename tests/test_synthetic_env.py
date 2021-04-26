#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# pytest tests/test_synthetic_env.py -s             #
#####################################################
import sys, random
import unittest
import pytest
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from datasets import ConstantFunc, ComposedSinFunc
from datasets import SyntheticDEnv


class TestSynethicEnv(unittest.TestCase):
    """Test the synethtic environment."""

    def test_simple(self):
        mean_generator = ComposedSinFunc(constant=0.1)
        std_generator = ConstantFunc(constant=0.5)

        dataset = SyntheticDEnv([mean_generator], [[std_generator]], num_per_task=5000)
        print(dataset)
        for timestamp, tau in dataset:
            assert tau.shape == (5000, 1)
