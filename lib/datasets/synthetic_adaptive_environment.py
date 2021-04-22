#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
import numpy as np
from typing import Optional
import torch
import torch.utils.data as data


class QuadraticFunction:
    """The quadratic function that outputs f(x) = a * x^2 + b * x + c."""

    def __init__(self, list_of_points=None):
        self._params = dict(a=None, b=None, c=None)
        if list_of_points is not None:
            self.fit(list_of_points)

    def set(self, a, b, c):
        self._params["a"] = a
        self._params["b"] = b
        self._params["c"] = c

    def check_valid(self):
        for key, value in self._params.items():
            if value is None:
                raise ValueError("The {:} is None".format(key))

    def __getitem__(self, x):
        self.check_valid()
        return self._params["a"] * x * x + self._params["b"] * x + self._params["c"]

    def fit(
        self,
        list_of_points,
        transf=lambda x: x,
        max_iter=900,
        lr_max=1.0,
        verbose=False,
    ):
        with torch.no_grad():
            data = torch.Tensor(list_of_points).type(torch.float32)
            assert data.ndim == 2 and data.size(1) == 2, "Invalid shape : {:}".format(
                data.shape
            )
            x, y = data[:, 0], data[:, 1]
        weights = torch.nn.Parameter(torch.Tensor(3))
        torch.nn.init.normal_(weights, mean=0.0, std=1.0)
        optimizer = torch.optim.Adam([weights], lr=lr_max, amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(max_iter*0.25), int(max_iter*0.5), int(max_iter*0.75)], gamma=0.1)
        if verbose:
            print("The optimizer: {:}".format(optimizer))

        best_loss = None
        for _iter in range(max_iter):
            y_hat = transf(weights[0] * x * x + weights[1] * x + weights[2])
            loss = torch.mean(torch.abs(y - y_hat))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if verbose:
                print(
                    "In QuadraticFunction's fit, loss at the {:02d}/{:02d}-th iter is {:}".format(
                        _iter, max_iter, loss.item()
                    )
                )
            # Update the params
            if best_loss is None or best_loss > loss.item():
                best_loss = loss.item()
                self._params["a"] = weights[0].item()
                self._params["b"] = weights[1].item()
                self._params["c"] = weights[2].item()

    def __repr__(self):
        return "{name}(y = {a} * x^2 + {b} * x + {c})".format(
            name=self.__class__.__name__,
            a=self._params["a"],
            b=self._params["b"],
            c=self._params["c"],
        )


class SynAdaptiveEnv(data.Dataset):
    """The synethtic dataset for adaptive environment.

    - x in [0, 1]
    - y = amplitude-scale-of(x) * sin( period-phase-shift-of(x) )
    - where
    - the amplitude scale is a quadratic function of x
    - the period-phase-shift is another quadratic function of x

    """

    def __init__(
        self,
        num: int = 100,
        num_sin_phase: int = 4,
        min_amplitude: float = 1,
        max_amplitude: float = 4,
        phase_shift: float = 0,
        mode: Optional[str] = None,
    ):
        self._amplitude_scale = QuadraticFunction(
            [(0, min_amplitude), (0.5, max_amplitude), (0, min_amplitude)]
        )

        self._num_sin_phase = num_sin_phase
        self._interval = 1.0 / (float(num) - 1)
        self._total_num = num

        self._period_phase_shift = QuadraticFunction()

        fitting_data = []
        temp_max_scalar = 2 ** num_sin_phase
        for i in range(num_sin_phase):
            value = (2 ** i) / temp_max_scalar
            fitting_data.append((value, math.sin(value)))
        self._period_phase_shift.fit(fitting_data, transf=lambda x: torch.sin(x))

        # Training Set 60%
        num_of_train = int(self._total_num * 0.6)
        # Validation Set 20%
        num_of_valid = int(self._total_num * 0.2)
        # Test Set 20%
        num_of_set = self._total_num - num_of_train - num_of_valid
        all_indexes = list(range(self._total_num))
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
        # transformation function
        self._transform = None

    def set_transform(self, fn):
        self._transform = fn

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
        position = self._interval * index
        value = self._amplitude_scale[position] * math.sin(
            self._period_phase_shift[position]
        )
        return index, position, value

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements)".format(
            name=self.__class__.__name__, cur_num=self._total_num, total=len(self)
        )
