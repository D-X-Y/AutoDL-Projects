#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################

import abc
import math
import random

from typing import Optional


class Space(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def random(self, recursion=True):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError


class Categorical(Space):
    def __init__(self, *data, default: Optional[int] = None):
        self._candidates = [*data]
        self._default = default
        assert self._default is None or 0 <= self._default < len(self._candidates), "default >= {:}".format(
            len(self._candidates)
        )

    def __getitem__(self, index):
        return self._candidates[index]

    def __len__(self):
        return len(self._candidates)

    def __repr__(self):
        return "{name:}(candidates={cs:}, default_index={default:})".format(
            name=self.__class__.__name__, cs=self._candidates, default=self._default
        )

    def random(self, recursion=True):
        sample = random.choice(self._candidates)
        if recursion and isinstance(sample, Space):
            return sample.random(recursion)
        else:
            return sample


class Continuous(Space):
    def __init__(self, lower: float, upper: float, default: Optional[float] = None, log: bool = False):
        self._lower = lower
        self._upper = upper
        self._default = default
        self._log_scale = log

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def default(self):
        return self._default

    def __repr__(self):
        return "{name:}(lower={lower:}, upper={upper:}, default_value={default:}, log_scale={log:})".format(
            name=self.__class__.__name__,
            lower=self._lower,
            upper=self._upper,
            default=self._default,
            log=self._log_scale,
        )

    def random(self, recursion=True):
        del recursion
        if self._log_scale:
            sample = random.uniform(math.log(self._lower), math.log(self._upper))
            return math.exp(sample)
        else:
            return random.uniform(self._lower, self._upper)
