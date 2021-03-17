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


class TestBasicSpace(unittest.TestCase):
    def test_categorical(self):
        space = Categorical(1, 2, 3, 4)
        for i in range(4):
            self.assertEqual(space[i], i + 1)
        self.assertEqual("Categorical(candidates=[1, 2, 3, 4])", str(space))
