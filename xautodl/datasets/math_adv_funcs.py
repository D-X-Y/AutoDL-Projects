#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
import abc
import copy
import numpy as np
from typing import Optional
import torch
import torch.utils.data as data

from .math_base_funcs import FitFunc
from .math_base_funcs import QuadraticFunc
from .math_base_funcs import QuarticFunc


class ConstantFunc(FitFunc):
    """The constant function: f(x) = c."""

    def __init__(self, constant=None):
        param = dict()
        param[0] = constant
        super(ConstantFunc, self).__init__(0, None, param)

    def __call__(self, x):
        self.check_valid()
        return self._params[0]

    def fit(self, **kwargs):
        raise NotImplementedError

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "{name}({a})".format(name=self.__class__.__name__, a=self._params[0])


class ComposedSinFunc(FitFunc):
    """The composed sin function that outputs:
      f(x) = amplitude-scale-of(x) * sin( period-phase-shift-of(x) )
    - the amplitude scale is a quadratic function of x
    - the period-phase-shift is another quadratic function of x
    """

    def __init__(self, **kwargs):
        super(ComposedSinFunc, self).__init__(0, None)
        self.fit(**kwargs)

    def __call__(self, x):
        self.check_valid()
        scale = self._params["amplitude_scale"](x)
        period_phase = self._params["period_phase_shift"](x)
        return scale * math.sin(period_phase)

    def fit(self, **kwargs):
        num_sin_phase = kwargs.get("num_sin_phase", 7)
        sin_speed_use_power = kwargs.get("sin_speed_use_power", True)
        min_amplitude = kwargs.get("min_amplitude", 1)
        max_amplitude = kwargs.get("max_amplitude", 4)
        phase_shift = kwargs.get("phase_shift", 0.0)
        # create parameters
        if kwargs.get("amplitude_scale", None) is None:
            amplitude_scale = QuadraticFunc(
                [(0, min_amplitude), (0.5, max_amplitude), (1, min_amplitude)]
            )
        else:
            amplitude_scale = kwargs.get("amplitude_scale")
        if kwargs.get("period_phase_shift", None) is None:
            fitting_data = []
            if sin_speed_use_power:
                temp_max_scalar = 2 ** (num_sin_phase - 1)
            else:
                temp_max_scalar = num_sin_phase - 1
            for i in range(num_sin_phase):
                if sin_speed_use_power:
                    value = (2 ** i) / temp_max_scalar
                    next_value = (2 ** (i + 1)) / temp_max_scalar
                else:
                    value = i / temp_max_scalar
                    next_value = (i + 1) / temp_max_scalar
                for _phase in (0, 0.25, 0.5, 0.75):
                    inter_value = value + (next_value - value) * _phase
                    fitting_data.append((inter_value, math.pi * (2 * i + _phase)))
            period_phase_shift = QuarticFunc(fitting_data)
        else:
            period_phase_shift = kwargs.get("period_phase_shift")
        self.set(
            dict(amplitude_scale=amplitude_scale, period_phase_shift=period_phase_shift)
        )

    def _getitem(self, x, weights):
        raise NotImplementedError

    def __repr__(self):
        return "{name}({amplitude_scale} * sin({period_phase_shift}))".format(
            name=self.__class__.__name__,
            amplitude_scale=self._params["amplitude_scale"],
            period_phase_shift=self._params["period_phase_shift"],
        )
