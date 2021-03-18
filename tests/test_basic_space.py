#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import sys, random
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
    """Test the basic search spaces."""

    def test_categorical(self):
        space = Categorical(1, 2, 3, 4)
        for i in range(4):
            self.assertEqual(space[i], i + 1)
        self.assertEqual(
            "Categorical(candidates=[1, 2, 3, 4], default_index=None)", str(space)
        )

    def test_continuous(self):
        random.seed(999)
        space = Continuous(0, 1)
        self.assertGreaterEqual(space.random(), 0)
        self.assertGreaterEqual(1, space.random())

        lower, upper = 1.5, 4.6
        space = Continuous(lower, upper, log=False)
        values = []
        for i in range(1000000):
            x = space.random()
            self.assertGreaterEqual(x, lower)
            self.assertGreaterEqual(upper, x)
            values.append(x)
        self.assertAlmostEqual((lower + upper) / 2, sum(values) / len(values), places=2)
        self.assertEqual(
            "Continuous(lower=1.5, upper=4.6, default_value=None, log_scale=False)",
            str(space),
        )

    def test_determined_and_has(self):
        # Test Non-nested Space
        space = Categorical(1, 2, 3, 4)
        self.assertFalse(space.determined)
        self.assertTrue(space.has(2))
        self.assertFalse(space.has(6))
        space = Categorical(4)
        self.assertTrue(space.determined)

        space = Continuous(0.11, 0.12)
        self.assertTrue(space.has(0.115))
        self.assertFalse(space.has(0.1))
        self.assertFalse(space.determined)
        space = Continuous(0.11, 0.11)
        self.assertTrue(space.determined)

        # Test Nested Space
        space_1 = Categorical(1, 2, 3, 4)
        space_2 = Categorical(1)
        nested_space = Categorical(space_1)
        self.assertFalse(nested_space.determined)
        self.assertTrue(nested_space.has(4))
        nested_space = Categorical(space_2)
        self.assertTrue(nested_space.determined)

        # Test Nested Space 2
        nested_space = Categorical(
            Categorical(1, 2, 3),
            Categorical(4, Categorical(5, 6, 7, Categorical(8, 9), 10), 11),
            12,
        )
        for i in range(1, 13):
            self.assertTrue(nested_space.has(i))
