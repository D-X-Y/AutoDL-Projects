#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import abc
import numpy as np


def assert_list_tuple(x):
    assert isinstance(x, (list, tuple))
    return len(x)


class DynamicGenerator(abc.ABC):
    """The dynamic quadratic function, where each param is a function."""

    def __init__(self):
        self._ndim = None

    def __call__(self, time, num):
        raise NotImplementedError


class UniformDGenerator(DynamicGenerator):
    """Generate data from the uniform distribution."""

    def __init__(self, l_functors, r_functors):
        super(UniformDGenerator, self).__init__()
        self._ndim = assert_list_tuple(l_functors)
        assert self._ndim == assert_list_tuple(r_functors)
        self._l_functors = l_functors
        self._r_functors = r_functors

    @property
    def ndim(self):
        return self._ndim

    def output_shape(self):
        return (self._ndim,)

    def __call__(self, time, num):
        l_list = [functor(time) for functor in self._l_functors]
        r_list = [functor(time) for functor in self._r_functors]
        values = []
        for l, r in zip(l_list, r_list):
            values.append(np.random.uniform(low=l, high=r, size=num))
        return np.stack(values, axis=-1)

    def __repr__(self):
        return "{name}({ndim} dims)".format(
            name=self.__class__.__name__, ndim=self._ndim
        )


class GaussianDGenerator(DynamicGenerator):
    """Generate data from Gaussian distribution."""

    def __init__(self, mean_functors, cov_functors, trunc=(-1, 1)):
        super(GaussianDGenerator, self).__init__()
        self._ndim = assert_list_tuple(mean_functors)
        assert self._ndim == len(
            cov_functors
        ), "length does not match {:} vs. {:}".format(self._ndim, len(cov_functors))
        assert_list_tuple(cov_functors)
        for cov_functor in cov_functors:
            assert self._ndim == assert_list_tuple(
                cov_functor
            ), "length does not match {:} vs. {:}".format(self._ndim, len(cov_functor))
        assert (
            isinstance(trunc, (list, tuple)) and len(trunc) == 2 and trunc[0] < trunc[1]
        )
        self._mean_functors = mean_functors
        self._cov_functors = cov_functors
        if trunc is not None:
            assert assert_list_tuple(trunc) == 2 and trunc[0] < trunc[1]
        self._trunc = trunc

    @property
    def ndim(self):
        return self._ndim

    def output_shape(self):
        return (self._ndim,)

    def __call__(self, time, num):
        mean_list = [functor(time) for functor in self._mean_functors]
        cov_matrix = [
            [abs(cov_gen(time)) for cov_gen in cov_functor]
            for cov_functor in self._cov_functors
        ]
        values = np.random.multivariate_normal(mean_list, cov_matrix, size=num)
        if self._trunc is not None:
            np.clip(values, self._trunc[0], self._trunc[1], out=values)
        return values

    def __repr__(self):
        return "{name}({ndim} dims, trunc={trunc})".format(
            name=self.__class__.__name__, ndim=self._ndim, trunc=self._trunc
        )
