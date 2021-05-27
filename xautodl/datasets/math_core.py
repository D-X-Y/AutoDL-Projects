#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.05 #
#####################################################
from .math_static_funcs import (
    LinearSFunc,
    QuadraticSFunc,
    CubicSFunc,
    QuarticSFunc,
    ConstantFunc,
    ComposedSinSFunc,
    ComposedCosSFunc,
)
from .math_dynamic_funcs import (
    LinearDFunc,
    QuadraticDFunc,
    SinQuadraticDFunc,
    BinaryQuadraticDFunc,
)
from .math_dynamic_generator import UniformDGenerator, GaussianDGenerator
