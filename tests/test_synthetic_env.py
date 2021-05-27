#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# pytest tests/test_synthetic_env.py -s             #
#####################################################
import unittest

from xautodl.datasets.synthetic_core import get_synthetic_env


class TestSynethicEnv(unittest.TestCase):
    """Test the synethtic environment."""

    def test_simple(self):
        versions = ["v1", "v2", "v3", "v4"]
        for version in versions:
            env = get_synthetic_env(version=version)
            print(env)
            for timestamp, (x, y) in env:
                self.assertEqual(x.shape, (1000, env._data_generator.ndim))
