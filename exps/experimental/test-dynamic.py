#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
# python test-dynamic.py
#####################################################
import sys
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "..").resolve()
print("LIB-DIR: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from xautodl.datasets.math_core import ConstantFunc
from xautodl.datasets.math_core import GaussianDGenerator

mean_generator = ConstantFunc(0)
cov_generator = ConstantFunc(1)

generator = GaussianDGenerator([mean_generator], [[cov_generator]], (-1, 1))
generator(0, 10)
