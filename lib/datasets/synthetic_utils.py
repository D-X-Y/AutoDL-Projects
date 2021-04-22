#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
import abc
import numpy as np
from typing import Optional
import torch
import torch.utils.data as data

from .math_base_funcs import QuadraticFunc, QuarticFunc


class UnifiedSplit:
    """A class to unify the split strategy."""

    def __init__(self, total_num, mode):
        # Training Set 60%
        num_of_train = int(total_num * 0.6)
        # Validation Set 20%
        num_of_valid = int(total_num * 0.2)
        # Test Set 20%
        num_of_set = total_num - num_of_train - num_of_valid
        all_indexes = list(range(total_num))
        if mode is None:
            self._indexes = all_indexes
        elif mode.lower() in ("train", "training"):
            self._indexes = all_indexes[:num_of_train]
        elif mode.lower() in ("valid", "validation"):
            self._indexes = all_indexes[num_of_train : num_of_train + num_of_valid]
        elif mode.lower() in ("test", "testing"):
            self._indexes = all_indexes[num_of_train + num_of_valid :]
        else:
            raise ValueError("Unkonwn mode of {:}".format(mode))
        self._mode = mode

    @property
    def mode(self):
        return self._mode


class SinGenerator(UnifiedSplit, data.Dataset):
    """The synethtic generator for the dynamically changing environment.

    - x in [0, 1]
    - y = amplitude-scale-of(x) * sin( period-phase-shift-of(x) )
    - where
    - the amplitude scale is a quadratic function of x
    - the period-phase-shift is another quadratic function of x

    """

    def __init__(
        self,
        num: int = 100,
        num_sin_phase: int = 7,
        min_amplitude: float = 1,
        max_amplitude: float = 4,
        phase_shift: float = 0,
        mode: Optional[str] = None,
    ):
        self._amplitude_scale = QuadraticFunc(
            [(0, min_amplitude), (0.5, max_amplitude), (1, min_amplitude)]
        )

        self._num_sin_phase = num_sin_phase
        self._interval = 1.0 / (float(num) - 1)
        self._total_num = num

        fitting_data = []
        temp_max_scalar = 2 ** (num_sin_phase - 1)
        for i in range(num_sin_phase):
            value = (2 ** i) / temp_max_scalar
            next_value = (2 ** (i + 1)) / temp_max_scalar
            for _phase in (0, 0.25, 0.5, 0.75):
                inter_value = value + (next_value - value) * _phase
                fitting_data.append((inter_value, math.pi * (2 * i + _phase)))
        self._period_phase_shift = QuarticFunc(fitting_data)
        UnifiedSplit.__init__(self, self._total_num, mode)
        self._transform = None

    def __iter__(self):
        self._iter_num = 0
        return self

    def __next__(self):
        if self._iter_num >= len(self):
            raise StopIteration
        self._iter_num += 1
        return self.__getitem__(self._iter_num - 1)

    def set_transform(self, transform):
        self._transform = transform

    def transform(self, x):
        if self._transform is None:
            return x
        else:
            return self._transform(x)

    def __getitem__(self, index):
        assert 0 <= index < len(self), "{:} is not in [0, {:})".format(index, len(self))
        index = self._indexes[index]
        position = self._interval * index
        value = self._amplitude_scale(position) * math.sin(
            self._period_phase_shift(position)
        )
        return index, position, self.transform(value)

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        return (
            "{name}({cur_num:}/{total} elements,\n"
            "amplitude={amplitude},\n"
            "period_phase_shift={period_phase_shift})".format(
                name=self.__class__.__name__,
                cur_num=len(self),
                total=self._total_num,
                amplitude=self._amplitude_scale,
                period_phase_shift=self._period_phase_shift,
            )
        )


class ConstantGenerator(UnifiedSplit, data.Dataset):
    """The constant generator."""

    def __init__(
        self,
        num: int = 100,
        constant: float = 0.1,
        mode: Optional[str] = None,
    ):
        self._total_num = num
        self._constant = constant
        UnifiedSplit.__init__(self, self._total_num, mode)

    def __iter__(self):
        self._iter_num = 0
        return self

    def __next__(self):
        if self._iter_num >= len(self):
            raise StopIteration
        self._iter_num += 1
        return self.__getitem__(self._iter_num - 1)

    def __getitem__(self, index):
        assert 0 <= index < len(self), "{:} is not in [0, {:})".format(index, len(self))
        index = self._indexes[index]
        return index, index, self._constant

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements)".format(
            name=self.__class__.__name__,
            cur_num=len(self),
            total=self._total_num,
        )
