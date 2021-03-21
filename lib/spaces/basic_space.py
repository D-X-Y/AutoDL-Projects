#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################

import abc
import math
import copy
import random
import numpy as np
from collections import OrderedDict

from typing import Optional, Text


__all__ = ["_EPS", "Space", "Categorical", "Integer", "Continuous"]

_EPS = 1e-9


class Space(metaclass=abc.ABCMeta):
    """Basic search space describing the set of possible candidate values for hyperparameter.
    All search space must inherit from this basic class.
    """

    def __init__(self):
        # used to avoid duplicate sample
        self._last_sample = None
        self._last_abstract = None

    @abc.abstractproperty
    def xrepr(self, depth=0) -> Text:
        raise NotImplementedError

    def __repr__(self) -> Text:
        return self.xrepr()

    @abc.abstractproperty
    def abstract(self, reuse_last=False) -> "Space":
        raise NotImplementedError

    @abc.abstractmethod
    def random(self, recursion=True, reuse_last=False):
        raise NotImplementedError

    @abc.abstractmethod
    def clean_last_sample(self):
        raise NotImplementedError

    @abc.abstractmethod
    def clean_last_abstract(self):
        raise NotImplementedError

    def clean_last(self):
        self.clean_last_sample()
        self.clean_last_abstract()

    @abc.abstractproperty
    def determined(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def has(self, x) -> bool:
        """Check whether x is in this search space."""
        assert not isinstance(
            x, Space
        ), "The input value itself can not be a search space."

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def copy(self) -> "Space":
        return copy.deepcopy(self)


class VirtualNode(Space):
    """For a nested search space, we represent it as a tree structure.

    For example,
    """

    def __init__(self, id=None, value=None):
        super(VirtualNode, self).__init__()
        self._id = id
        self._value = value
        self._attributes = OrderedDict()

    @property
    def value(self):
        return self._value

    def append(self, key, value):
        if not isinstance(key, str):
            raise TypeError(
                "Only accept string as a key instead of {:}".format(type(key))
            )
        if not isinstance(value, Space):
            raise ValueError("Invalid type of value: {:}".format(type(value)))
        # if value.determined:
        #    raise ValueError("Can not attach a determined value: {:}".format(value))
        self._attributes[key] = value

    def xrepr(self, depth=0) -> Text:
        strs = [self.__class__.__name__ + "(value={:}".format(self._value)]
        for key, value in self._attributes.items():
            strs.append(key + " = " + value.xrepr(depth + 1))
        strs.append(")")
        if len(strs) == 2:
            return "".join(strs)
        else:
            space = "  "
            xstrs = (
                [strs[0]]
                + [space * (depth + 1) + x for x in strs[1:-1]]
                + [space * depth + strs[-1]]
            )
            return ",\n".join(xstrs)

    def abstract(self, reuse_last=False) -> Space:
        if reuse_last and self._last_abstract is not None:
            return self._last_abstract
        node = VirtualNode(id(self))
        for key, value in self._attributes.items():
            if not value.determined:
                node.append(value.abstract(reuse_last))
        self._last_abstract = node
        return self._last_abstract

    def random(self, recursion=True, reuse_last=False):
        if reuse_last and self._last_sample is not None:
            return self._last_sample
        node = VirtualNode(None, self._value)
        for key, value in self._attributes.items():
            node.append(key, value.random(recursion, reuse_last))
        self._last_sample = node  # record the last sample
        return node

    def clean_last_sample(self):
        self._last_sample = None
        for key, value in self._attributes.items():
            value.clean_last_sample()

    def clean_last_abstract(self):
        self._last_abstract = None
        for key, value in self._attributes.items():
            value.clean_last_abstract()

    def has(self, x) -> bool:
        for key, value in self._attributes.items():
            if value.has(x):
                return True
        return False

    def __contains__(self, key):
        return key in self._attributes

    def __getitem__(self, key):
        return self._attributes[key]

    @property
    def determined(self) -> bool:
        for key, value in self._attributes.items():
            if not value.determined:
                return False
        return True

    def __eq__(self, other):
        if not isinstance(other, VirtualNode):
            return False
        for key, value in self._attributes.items():
            if not key in other:
                return False
            if value != other[key]:
                return False
        return True


class Categorical(Space):
    """A space contains the categorical values.
    It can be a nested space, which means that the candidate in this space can also be a search space.
    """

    def __init__(self, *data, default: Optional[int] = None):
        super(Categorical, self).__init__()
        self._candidates = [*data]
        self._default = default
        assert self._default is None or 0 <= self._default < len(
            self._candidates
        ), "default >= {:}".format(len(self._candidates))
        assert len(self) > 0, "Please provide at least one candidate"

    @property
    def candidates(self):
        return self._candidates

    @property
    def default(self):
        return self._default

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

    def clean_last_sample(self):
        self._last_sample = None
        for candidate in self._candidates:
            if isinstance(candidate, Space):
                candidate.clean_last_sample()

    def clean_last_abstract(self):
        self._last_abstract = None
        for candidate in self._candidates:
            if isinstance(candidate, Space):
                candidate.clean_last_abstract()

    def abstract(self, reuse_last=False) -> Space:
        if reuse_last and self._last_abstract is not None:
            return self._last_abstract
        if self.determined:
            result = VirtualNode(id(self), self)
        else:
            # [TO-IMPROVE]
            data = []
            for candidate in self.candidates:
                if isinstance(candidate, Space):
                    data.append(candidate.abstract())
                else:
                    data.append(VirtualNode(id(candidate), candidate))
            result = Categorical(*data, default=self._default)
        self._last_abstract = result
        return self._last_abstract

    def random(self, recursion=True, reuse_last=False):
        if reuse_last and self._last_sample is not None:
            return self._last_sample
        sample = random.choice(self._candidates)
        if recursion and isinstance(sample, Space):
            sample = sample.random(recursion, reuse_last)
        if isinstance(sample, VirtualNode):
            sample = sample.copy()
        else:
            sample = VirtualNode(None, sample)
        self._last_sample = sample
        return self._last_sample

    def xrepr(self, depth=0):
        del depth
        xrepr = "{name:}(candidates={cs:}, default_index={default:})".format(
            name=self.__class__.__name__, cs=self._candidates, default=self._default
        )
        return xrepr

    def has(self, x):
        super().has(x)
        for candidate in self._candidates:
            if isinstance(candidate, Space) and candidate.has(x):
                return True
            elif candidate == x:
                return True
        return False

    def __eq__(self, other):
        if not isinstance(other, Categorical):
            return False
        if len(self) != len(other):
            return False
        if self.default != other.default:
            return False
        for index in range(len(self)):
            if self.__getitem__(index) != other[index]:
                return False
        return True


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

    def xrepr(self, depth=0):
        del depth
        xrepr = "{name:}(lower={lower:}, upper={upper:}, default={default:})".format(
            name=self.__class__.__name__,
            lower=self._raw_lower,
            upper=self._raw_upper,
            default=self._raw_default,
        )
        return xrepr


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
        super(Continuous, self).__init__()
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
    def use_log(self):
        return self._log_scale

    @property
    def eps(self):
        return self._eps

    def abstract(self, reuse_last=False) -> Space:
        if reuse_last and self._last_abstract is not None:
            return self._last_abstract
        self._last_abstract = self.copy()
        return self._last_abstract

    def random(self, recursion=True, reuse_last=False):
        del recursion
        if reuse_last and self._last_sample is not None:
            return self._last_sample
        if self._log_scale:
            sample = random.uniform(math.log(self._lower), math.log(self._upper))
            sample = math.exp(sample)
        else:
            sample = random.uniform(self._lower, self._upper)
        self._last_sample = VirtualNode(None, sample)
        return self._last_sample

    def xrepr(self, depth=0):
        del depth
        xrepr = "{name:}(lower={lower:}, upper={upper:}, default_value={default:}, log_scale={log:})".format(
            name=self.__class__.__name__,
            lower=self._lower,
            upper=self._upper,
            default=self._default,
            log=self._log_scale,
        )
        return xrepr

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

    @property
    def determined(self):
        return abs(self.lower - self.upper) <= self._eps

    def clean_last_sample(self):
        self._last_sample = None

    def clean_last_abstract(self):
        self._last_abstract = None

    def __eq__(self, other):
        if not isinstance(other, Continuous):
            return False
        if self is other:
            return True
        else:
            return (
                self.lower == other.lower
                and self.upper == other.upper
                and self.default == other.default
                and self.use_log == other.use_log
                and self.eps == other.eps
            )
