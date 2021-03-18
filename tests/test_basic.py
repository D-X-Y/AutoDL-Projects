#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import sys
import unittest
import pytest
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from spaces import Categorical
from spaces import Continuous


class TestBasicSpace(unittest.TestCase):
    def test_categorical(self):
        space = Categorical(1, 2, 3, 4)
        for i in range(4):
            self.assertEqual(space[i], i + 1)
        self.assertEqual("Categorical(candidates=[1, 2, 3, 4], default_index=None)", str(space))

    def test_continuous(self):
        space = Continuous(0, 1)
        self.assertGreaterEqual(space.random(), 0)
        self.assertGreaterEqual(1, space.random())

        lower, upper = 1.5, 4.6
        space = Continuous(lower, upper, log=False)
        values = []
        for i in range(100000):
            x = space.random()
            self.assertGreaterEqual(x, lower)
            self.assertGreaterEqual(upper, x)
            values.append(x)
        self.assertAlmostEqual((lower + upper) / 2, sum(values) / len(values), places=2)
