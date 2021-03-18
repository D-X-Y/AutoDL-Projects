#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################

import abc
import math
import copy
import random
import numpy as np

from typing import Optional

_EPS = 1e-9


class Space(metaclass=abc.ABCMeta):
    """Basic search space describing the set of possible candidate values for hyperparameter.
    All search space must inherit from this basic class.
    """

    @abc.abstractmethod
    def random(self, recursion=True):
        raise NotImplementedError

    @abc.abstractproperty
    def determined(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def has(self, x):
        """Check whether x is in this search space."""
        assert not isinstance(
            x, Space
        ), "The input value itself can not be a search space."

    def copy(self):
        return copy.deepcopy(self)


class Categorical(Space):
    """A space contains the categorical values.
    It can be a nested space, which means that the candidate in this space can also be a search space.
    """

    def __init__(self, *data, default: Optional[int] = None):
        self._candidates = [*data]
        self._default = default
        assert self._default is None or 0 <= self._default < len(
            self._candidates
        ), "default >= {:}".format(len(self._candidates))
        assert len(self) > 0, "Please provide at least one candidate"

    @property
    def determined(self):
        if len(self) == 1:
            return (
                not isinstance(self._candidates[0], Space)
                or self._candidates[0].determined
            )
        else:
            return False

    def __getitem__(self, index):
        return self._candidates[index]

    def __len__(self):
        return len(self._candidates)

    def __repr__(self):
        return "{name:}(candidates={cs:}, default_index={default:})".format(
            name=self.__class__.__name__, cs=self._candidates, default=self._default
        )

    def has(self, x):
        super().has(x)
        for candidate in self._candidates:
            if isinstance(candidate, Space) and candidate.has(x):
                return True
            elif candidate == x:
                return True
        return False

    def random(self, recursion=True):
        sample = random.choice(self._candidates)
        if recursion and isinstance(sample, Space):
            return sample.random(recursion)
        else:
            return sample


class Integer(Categorical):
    """A space contains the integer values."""

    def __init__(self, lower: int, upper: int, default: Optional[int] = None):
        if not isinstance(lower, int) or not isinstance(upper, int):
            raise ValueError(
                "The lower [{:}] and uppwer [{:}] must be int.".format(lower, upper)
            )
        data = list(range(lower, upper + 1))
        self._raw_lower = lower
        self._raw_upper = upper
        self._raw_default = default
        if default is not None and (default < lower or default > upper):
            raise ValueError("The default value [{:}] is out of range.".format(default))
            default = data.index(default)
        super(Integer, self).__init__(*data, default=default)

    def __repr__(self):
        return "{name:}(lower={lower:}, upper={upper:}, default={default:})".format(
            name=self.__class__.__name__,
            lower=self._raw_lower,
            upper=self._raw_upper,
            default=self._raw_default,
        )


np_float_types = (np.float16, np.float32, np.float64)
np_int_types = (
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
)


class Continuous(Space):
    """A space contains the continuous values."""

    def __init__(
        self,
        lower: float,
        upper: float,
        default: Optional[float] = None,
        log: bool = False,
        eps: float = _EPS,
    ):
        self._lower = lower
        self._upper = upper
        self._default = default
        self._log_scale = log
        self._eps = eps

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def default(self):
        return self._default

    @property
    def determined(self):
        return abs(self.lower - self.upper) <= self._eps

    def __repr__(self):
        return "{name:}(lower={lower:}, upper={upper:}, default_value={default:}, log_scale={log:})".format(
            name=self.__class__.__name__,
            lower=self._lower,
            upper=self._upper,
            default=self._default,
            log=self._log_scale,
        )

    def convert(self, x):
        if isinstance(x, np_float_types) and x.size == 1:
            return float(x), True
        elif isinstance(x, np_int_types) and x.size == 1:
            return float(x), True
        elif isinstance(x, int):
            return float(x), True
        elif isinstance(x, float):
            return float(x), True
        else:
            return None, False

    def has(self, x):
        super().has(x)
        converted_x, success = self.convert(x)
        return success and self.lower <= converted_x <= self.upper

    def random(self, recursion=True):
        del recursion
        if self._log_scale:
            sample = random.uniform(math.log(self._lower), math.log(self._upper))
            return math.exp(sample)
        else:
            return random.uniform(self._lower, self._upper)
