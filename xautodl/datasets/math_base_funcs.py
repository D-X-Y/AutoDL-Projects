#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
import abc
import copy
import numpy as np


class MathFunc(abc.ABC):
    """The math function -- a virtual class defining some APIs."""

    def __init__(self, freedom: int, params=None, xstr="x"):
        # initialize as empty
        self._params = dict()
        for i in range(freedom):
            self._params[i] = None
        self._freedom = freedom
        if params is not None:
            self.set(params)
        self._xstr = str(xstr)
        self._skip_check = True

    def set(self, params):
        for key in range(self._freedom):
            param = copy.deepcopy(params[key])
            self._params[key] = param

    def check_valid(self):
        if not self._skip_check:
            for key in range(self._freedom):
                value = self._params[key]
                if value is None:
                    raise ValueError("The {:} is None".format(key))

    @property
    def xstr(self):
        return self._xstr

    def reset_xstr(self, xstr):
        self._xstr = str(xstr)

    def output_shape(self, input_shape):
        return input_shape

    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @abc.abstractmethod
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
