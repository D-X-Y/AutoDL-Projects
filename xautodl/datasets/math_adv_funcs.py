#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
import abc
import copy
import numpy as np
from typing import Optional
import torch
import torch.utils.data as data

from .math_base_funcs import FitFunc
from .math_base_funcs import QuadraticFunc
from .math_base_funcs import QuarticFunc


class ConstantFunc(FitFunc):
    """The constant function: f(x) = c."""

    def __init__(self, constant=None, xstr="x"):
        param = dict()
        param[0] = constant
        super(ConstantFunc, self).__init__(0, None, param, xstr)

    def __call__(self, x):
        self.check_valid()
        return self._params[0]

    def fit(self, **kwargs):
        raise NotImplementedError

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "{name}({a})".format(name=self.__class__.__name__, a=self._params[0])


class ComposedSinFunc(FitFunc):
    """The composed sin function that outputs:
    f(x) = a * sin( b*x ) + c
    """

    def __init__(self, params, xstr="x"):
        super(ComposedSinFunc, self).__init__(3, None, params, xstr)

    def __call__(self, x):
        self.check_valid()
        a = self._params[0]
        b = self._params[1]
        c = self._params[2]
        return a * math.sin(b * x) + c

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "{name}({a} * sin({b} * {x}) + {c})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )


class ComposedCosFunc(FitFunc):
    """The composed sin function that outputs:
    f(x) = a * cos( b*x ) + c
    """

    def __init__(self, params, xstr="x"):
        super(ComposedCosFunc, self).__init__(3, None, params, xstr)

    def __call__(self, x):
        self.check_valid()
        a = self._params[0]
        b = self._params[1]
        c = self._params[2]
        return a * math.cos(b * x) + c

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "{name}({a} * sin({b} * {x}) + {c})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )
