#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# pytest tests/test_synthetic_env.py -s             #
#####################################################
import unittest

from xautodl.datasets.math_core import ConstantFunc, ComposedSinFunc
from xautodl.datasets.synthetic_core import SyntheticDEnv


class TestSynethicEnv(unittest.TestCase):
    """Test the synethtic environment."""

    def test_simple(self):
        mean_generator = ComposedSinFunc(constant=0.1)
        std_generator = ConstantFunc(constant=0.5)
        dataset = SyntheticDEnv([mean_generator], [[std_generator]], num_per_task=5000)
        print(dataset)
        for timestamp, tau in dataset:
            self.assertEqual(tau.shape, (5000, 1))

    def test_length(self):
        mean_generator = ComposedSinFunc(constant=0.1)
        std_generator = ConstantFunc(constant=0.5)
        dataset = SyntheticDEnv([mean_generator], [[std_generator]], num_per_task=5000)
        self.assertEqual(len(dataset), 100)

        dataset = SyntheticDEnv([mean_generator], [[std_generator]], mode="train")
        self.assertEqual(len(dataset), 60)
