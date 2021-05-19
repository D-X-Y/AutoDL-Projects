#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
from .basic_space import Space
from .basic_space import VirtualNode
from .basic_space import Integer
from .basic_space import Continuous
from .basic_space import Categorical
from .basic_space import _EPS


def has_categorical(space_or_value, x):
    if isinstance(space_or_value, Space):
        return space_or_value.has(x)
    else:
        return space_or_value == x


def has_continuous(space_or_value, x):
    if isinstance(space_or_value, Space):
        return space_or_value.has(x)
    else:
        return abs(space_or_value - x) <= _EPS


def is_determined(space_or_value):
    if isinstance(space_or_value, Space):
        return space_or_value.determined
    else:
        return True


def get_determined_value(space_or_value):
    if not is_determined(space_or_value):
        raise ValueError("This input is not determined: {:}".format(space_or_value))
    if isinstance(space_or_value, Space):
        if isinstance(space_or_value, Continuous):
            return space_or_value.lower
        elif isinstance(space_or_value, Categorical):
            return get_determined_value(space_or_value[0])
        else:  # VirtualNode
            return space_or_value.value
    else:
        return space_or_value


def get_max(space_or_value):
    if isinstance(space_or_value, Integer):
        return max(space_or_value.candidates)
    elif isinstance(space_or_value, Continuous):
        return space_or_value.upper
    elif isinstance(space_or_value, Categorical):
        values = []
        for index in range(len(space_or_value)):
            max_value = get_max(space_or_value[index])
            values.append(max_value)
        return max(values)
    else:
        return space_or_value


def get_min(space_or_value):
    if isinstance(space_or_value, Integer):
        return min(space_or_value.candidates)
    elif isinstance(space_or_value, Continuous):
        return space_or_value.lower
    elif isinstance(space_or_value, Categorical):
        values = []
        for index in range(len(space_or_value)):
            min_value = get_min(space_or_value[index])
            values.append(min_value)
        return min(values)
    else:
        return space_or_value
