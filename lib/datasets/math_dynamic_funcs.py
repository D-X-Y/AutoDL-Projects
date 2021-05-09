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


class DynamicFunc(FitFunc):
    """The dynamic quadratic function, where each param is a function."""

    def __init__(self, freedom: int, params=None):
        super(DynamicFunc, self).__init__(freedom, None, params)
        self._timestamp = None

    def __call__(self, x, timestamp=None):
        raise NotImplementedError

    def _getitem(self, x, weights):
        raise NotImplementedError

    def set_timestamp(self, timestamp):
        self._timestamp = timestamp

    def noise_call(self, x, timestamp=None, std=0.1):
        clean_y = self.__call__(x, timestamp)
        if isinstance(clean_y, np.ndarray):
            noise_y = clean_y + np.random.normal(scale=std, size=clean_y.shape)
        else:
            raise ValueError("Unkonwn type: {:}".format(type(clean_y)))
        return noise_y


class DynamicLinearFunc(DynamicFunc):
    """The dynamic linear function that outputs f(x) = a * x + b.
    The a and b is a function of timestamp.
    """

    def __init__(self, params=None):
        super(DynamicLinearFunc, self).__init__(3, params)

    def __call__(self, x, timestamp=None):
        self.check_valid()
        if timestamp is None:
            timestamp = self._timestamp
        a = self._params[0](timestamp)
        b = self._params[1](timestamp)
        convert_fn = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        a, b = convert_fn(a), convert_fn(b)
        return a * x + b

    def __repr__(self):
        return "{name}({a} * x + {b}, timestamp={timestamp})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            timestamp=self._timestamp,
        )


class DynamicQuadraticFunc(DynamicFunc):
    """The dynamic quadratic function that outputs f(x) = a * x^2 + b * x + c.
    The a, b, and c is a function of timestamp.
    """

    def __init__(self, params=None):
        super(DynamicQuadraticFunc, self).__init__(3, params)

    def __call__(self, x, timestamp=None):
        self.check_valid()
        if timestamp is None:
            timestamp = self._timestamp
        a = self._params[0](timestamp)
        b = self._params[1](timestamp)
        c = self._params[2](timestamp)
        convert_fn = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        a, b, c = convert_fn(a), convert_fn(b), convert_fn(c)
        return a * x * x + b * x + c

    def __repr__(self):
        return "{name}({a} * x^2 + {b} * x + {c}, timestamp={timestamp})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            timestamp=self._timestamp,
        )
