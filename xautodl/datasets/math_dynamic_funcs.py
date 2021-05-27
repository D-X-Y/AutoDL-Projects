#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
import abc
import copy
import numpy as np

from .math_base_funcs import MathFunc


class DynamicFunc(MathFunc):
    """The dynamic function, where each param is a function."""

    def __init__(self, freedom: int, params=None, xstr="x"):
        if params is not None:
            for key, param in params.items():
                param.reset_xstr("t") if isinstance(param, MathFunc) else None
        super(DynamicFunc, self).__init__(freedom, params, xstr)

    def noise_call(self, x, timestamp, std):
        clean_y = self.__call__(x, timestamp)
        if std is None:
            noise_y = clean_y
        elif isinstance(clean_y, np.ndarray):
            noise_y = clean_y + np.random.normal(scale=std, size=clean_y.shape)
        else:
            raise ValueError("Unkonwn type: {:}".format(type(clean_y)))
        return noise_y


class LinearDFunc(DynamicFunc):
    """The dynamic linear function that outputs f(x) = a * x + b.
    The a and b is a function of timestamp.
    """

    def __init__(self, params, xstr="x"):
        super(LinearDFunc, self).__init__(2, params, xstr)

    def __call__(self, x, timestamp):
        a = self._params[0](timestamp)
        b = self._params[1](timestamp)
        convert_fn = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        a, b = convert_fn(a), convert_fn(b)
        return a * x + b

    def __repr__(self):
        return "({a} * {x} + {b})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            x=self.xstr,
        )


class QuadraticDFunc(DynamicFunc):
    """The dynamic quadratic function that outputs f(x) = a * x^2 + b * x + c.
    The a, b, and c is a function of timestamp.
    """

    def __init__(self, params, xstr="x"):
        super(QuadraticDFunc, self).__init__(3, params)

    def __call__(self, x, timestamp):
        self.check_valid()
        a = self._params[0](timestamp)
        b = self._params[1](timestamp)
        c = self._params[2](timestamp)
        convert_fn = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        a, b, c = convert_fn(a), convert_fn(b), convert_fn(c)
        return a * x * x + b * x + c

    def __repr__(self):
        return "({a} * {x}^2 + {b} * {x} + {c})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )


class SinQuadraticDFunc(DynamicFunc):
    """The dynamic quadratic function that outputs f(x) = sin(a * x^2 + b * x + c).
    The a, b, and c is a function of timestamp.
    """

    def __init__(self, params=None):
        super(SinQuadraticDFunc, self).__init__(3, params)

    def __call__(self, x, timestamp):
        self.check_valid()
        a = self._params[0](timestamp)
        b = self._params[1](timestamp)
        c = self._params[2](timestamp)
        convert_fn = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        a, b, c = convert_fn(a), convert_fn(b), convert_fn(c)
        return np.sin(a * x * x + b * x + c)

    def __repr__(self):
        return "{name}({a} * {x}^2 + {b} * {x} + {c})".format(
            name="Sin",
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )


class BinaryQuadraticDFunc(DynamicFunc):
    """The dynamic quadratic function that outputs f(x) = a * x[0]^2 + b * x[1] + c >= 0.
    The a, b, and c is a function of timestamp.
    """

    def __init__(self, params=None):
        super(BinaryQuadraticDFunc, self).__init__(3, params)

    def __call__(self, x, timestamp):
        self.check_valid()
        a = self._params[0](timestamp)
        b = self._params[1](timestamp)
        c = self._params[2](timestamp)
        convert_fn = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        a, b, c = convert_fn(a), convert_fn(b), convert_fn(c)
        if isinstance(x, np.ndarray) and x.shape[-1] == 2:
            results = a * x[..., 0] * x[..., 0] + b * x[..., 1] + c
            return (results >= 0).astype(np.int)
        else:
            raise ValueError(
                "Either the type {:} or the shape is incorrect.".format(type(x))
            )

    def __repr__(self):
        return "({a} * {x}[0]^2 + {b} * {x}[1] + {c} >= 0)".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )
