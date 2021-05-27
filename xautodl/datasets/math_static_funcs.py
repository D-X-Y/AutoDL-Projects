#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
import abc
import copy
import numpy as np

from .math_base_funcs import MathFunc


class StaticFunc(MathFunc):
    """The fit function that outputs f(x) = a * x^2 + b * x + c."""

    def __init__(self, freedom: int, params=None, xstr="x"):
        super(StaticFunc, self).__init__(freedom, params, xstr)

    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    def noise_call(self, x, std):
        clean_y = self.__call__(x)
        if isinstance(clean_y, np.ndarray):
            noise_y = clean_y + np.random.normal(scale=std, size=clean_y.shape)
        else:
            raise ValueError("Unkonwn type: {:}".format(type(clean_y)))
        return noise_y

    def __repr__(self):
        return "{name}(freedom={freedom})".format(
            name=self.__class__.__name__, freedom=freedom
        )


class LinearSFunc(StaticFunc):
    """The linear function that outputs f(x) = a * x + b."""

    def __init__(self, params=None, xstr="x"):
        super(LinearSFunc, self).__init__(2, params, xstr)

    def __call__(self, x):
        self.check_valid()
        return self._params[0] * x + self._params[1]

    def _getitem(self, x, weights):
        return weights[0] * x + weights[1]

    def __repr__(self):
        return "({a} * {x} + {b})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            x=self.xstr,
        )


class QuadraticSFunc(StaticFunc):
    """The quadratic function that outputs f(x) = a * x^2 + b * x + c."""

    def __init__(self, params=None, xstr="x"):
        super(QuadraticSFunc, self).__init__(3, params, xstr)

    def __call__(self, x):
        self.check_valid()
        return self._params[0] * x * x + self._params[1] * x + self._params[2]

    def _getitem(self, x, weights):
        return weights[0] * x * x + weights[1] * x + weights[2]

    def __repr__(self):
        return "({a} * {x}^2 + {b} * {x} + {c})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )


class CubicSFunc(StaticFunc):
    """The cubic function that outputs f(x) = a * x^3 + b * x^2 + c * x + d."""

    def __init__(self, params=None, xstr="x"):
        super(CubicSFunc, self).__init__(4, params, xstr)

    def __call__(self, x):
        self.check_valid()
        return (
            self._params[0] * x ** 3
            + self._params[1] * x ** 2
            + self._params[2] * x
            + self._params[3]
        )

    def _getitem(self, x, weights):
        return weights[0] * x ** 3 + weights[1] * x ** 2 + weights[2] * x + weights[3]

    def __repr__(self):
        return "({a} * {x}^3 + {b} * {x}^2 + {c} * {x} + {d})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            d=self._params[3],
            x=self.xstr,
        )


class QuarticSFunc(StaticFunc):
    """The quartic function that outputs f(x) = a * x^4 + b * x^3 + c * x^2 + d * x + e."""

    def __init__(self, params=None, xstr="x"):
        super(QuarticSFunc, self).__init__(5, params, xstr)

    def __call__(self, x):
        self.check_valid()
        return (
            self._params[0] * x ** 4
            + self._params[1] * x ** 3
            + self._params[2] * x ** 2
            + self._params[3] * x
            + self._params[4]
        )

    def _getitem(self, x, weights):
        return (
            weights[0] * x ** 4
            + weights[1] * x ** 3
            + weights[2] * x ** 2
            + weights[3] * x
            + weights[4]
        )

    def __repr__(self):
        return (
            "{name}({a} * {x}^4 + {b} * {x}^3 + {c} * {x}^2 + {d} * {x} + {e})".format(
                name=self.__class__.__name__,
                a=self._params[0],
                b=self._params[1],
                c=self._params[2],
                d=self._params[3],
                e=self._params[3],
                x=self.xstr,
            )
        )


### advanced functions


class ConstantFunc(StaticFunc):
    """The constant function: f(x) = c."""

    def __init__(self, constant, xstr="x"):
        super(ConstantFunc, self).__init__(1, {0: constant}, xstr)

    def __call__(self, x):
        self.check_valid()
        return self._params[0]

    def fit(self, **kwargs):
        raise NotImplementedError

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "{a}".format(name=self.__class__.__name__, a=self._params[0])


class ComposedSinSFunc(StaticFunc):
    """The composed sin function that outputs:
    f(x) = a * sin( b*x ) + c
    """

    def __init__(self, params, xstr="x"):
        super(ComposedSinSFunc, self).__init__(3, params, xstr)

    def __call__(self, x):
        self.check_valid()
        a = self._params[0]
        b = self._params[1]
        c = self._params[2]
        return a * math.sin(b * x) + c

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "({a} * sin({b} * {x}) + {c})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )


class ComposedCosSFunc(StaticFunc):
    """The composed sin function that outputs:
    f(x) = a * cos( b*x ) + c
    """

    def __init__(self, params, xstr="x"):
        super(ComposedCosSFunc, self).__init__(3, params, xstr)

    def __call__(self, x):
        self.check_valid()
        a = self._params[0]
        b = self._params[1]
        c = self._params[2]
        return a * math.cos(b * x) + c

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "({a} * sin({b} * {x}) + {c})".format(
            name=self.__class__.__name__,
            a=self._params[0],
            b=self._params[1],
            c=self._params[2],
            x=self.xstr,
        )
