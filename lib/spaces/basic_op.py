from spaces.basic_space import Space
from spaces.basic_space import _EPS


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
