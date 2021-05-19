#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest tests/test_synthetic_utils.py -s           #
#####################################################
import unittest

from xautodl.datasets.synthetic_core import TimeStamp


class TestTimeStamp(unittest.TestCase):
    """Test the timestamp generator."""

    def test_simple(self):
        for mode in (None, "train", "valid", "test"):
            generator = TimeStamp(0, 1)
            print(generator)
            for idx, (i, xtime) in enumerate(generator):
                self.assertTrue(i == idx)
                if idx == 0:
                    self.assertTrue(xtime == 0)
                if idx + 1 == len(generator):
                    self.assertTrue(abs(xtime - 1) < 1e-8)
