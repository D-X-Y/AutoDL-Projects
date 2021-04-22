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

from datasets import ConstantGenerator, SinGenerator
from datasets import SyntheticDEnv


class TestSynethicEnv(unittest.TestCase):
    """Test the synethtic environment."""

    def test_simple(self):
        mean_generator = SinGenerator()
        std_generator = ConstantGenerator(constant=0.5)

        dataset = SyntheticDEnv([mean_generator], [[std_generator]])
        print(dataset)
        for timestamp, tau in dataset:
            assert tau.shape == (5000, 1)
