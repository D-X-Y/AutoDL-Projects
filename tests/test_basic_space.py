#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_basic_space.py -s               #
#####################################################
import random
import unittest

from xautodl.spaces import Categorical
from xautodl.spaces import Continuous
from xautodl.spaces import Integer
from xautodl.spaces import is_determined
from xautodl.spaces import get_min
from xautodl.spaces import get_max


class TestBasicSpace(unittest.TestCase):
    """Test the basic search spaces."""

    def test_categorical(self):
        space = Categorical(1, 2, 3, 4)
        for i in range(4):
            self.assertEqual(space[i], i + 1)
        self.assertEqual(
            "Categorical(candidates=[1, 2, 3, 4], default_index=None)", str(space)
        )

    def test_integer(self):
        space = Integer(lower=1, upper=4)
        for i in range(4):
            self.assertEqual(space[i], i + 1)
        self.assertEqual("Integer(lower=1, upper=4, default=None)", str(space))
        self.assertEqual(get_max(space), 4)
        self.assertEqual(get_min(space), 1)

    def test_continuous(self):
        random.seed(999)
        space = Continuous(0, 1)
        self.assertGreaterEqual(space.random().value, 0)
        self.assertGreaterEqual(1, space.random().value)

        lower, upper = 1.5, 4.6
        space = Continuous(lower, upper, log=False)
        values = []
        for i in range(1000000):
            x = space.random(reuse_last=False).value
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
        print("\nThe nested search space:\n{:}".format(nested_space))
        for i in range(1, 13):
            self.assertTrue(nested_space.has(i))

        # Test Simple Op
        self.assertTrue(is_determined(1))
        self.assertFalse(is_determined(nested_space))

    def test_duplicate(self):
        space = Categorical(1, 2, 3, 4)
        x = space.random()
        for _ in range(100):
            self.assertEqual(x, space.random(reuse_last=True))


class TestAbstractSpace(unittest.TestCase):
    """Test the abstract search spaces."""

    def test_continous(self):
        print("")
        space = Continuous(0, 1)
        self.assertEqual(space, space.abstract())
        print("The abstract search space for Continuous: {:}".format(space.abstract()))

        space = Categorical(1, 2, 3)
        self.assertEqual(len(space.abstract()), 3)
        print(space.abstract())

        nested_space = Categorical(
            Categorical(1, 2, 3),
            Categorical(4, Categorical(5, 6, 7, Categorical(8, 9), 10), 11),
            12,
        )
        abstract_nested_space = nested_space.abstract()
        print("The abstract nested search space:\n{:}".format(abstract_nested_space))
