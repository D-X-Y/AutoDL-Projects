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

from datasets import SynAdaptiveEnv


class TestSynAdaptiveEnv(unittest.TestCase):
    """Test the synethtic adaptive environment."""

    def test_simple(self):
        dataset = SynAdaptiveEnv()
        for i, (idx, t, x) in enumerate(dataset):
            assert i == idx, "First loop: {:} vs {:}".format(i, idx)
        for i, (idx, t, x) in enumerate(dataset):
            assert i == idx, "Second loop: {:} vs {:}".format(i, idx)
