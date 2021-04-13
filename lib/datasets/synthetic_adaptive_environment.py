#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import numpy as np
from typing import Optional
import torch.utils.data as data


class SynAdaptiveEnv(data.Dataset):
    """The synethtic dataset for adaptive environment."""

    def __init__(
        self,
        max_num_phase: int = 100,
        interval: float = 0.1,
        max_scale: float = 4,
        offset_scale: float = 1.5,
        mode: Optional[str] = None,
    ):

        self._max_num_phase = max_num_phase
        self._interval = interval

        self._times = np.arange(0, np.pi * self._max_num_phase, self._interval)
        xmin, xmax = self._times.min(), self._times.max()
        self._inputs = []
        self._total_num = len(self._times)
        for i in range(self._total_num):
            scale = (i + 1.0) / self._total_num * max_scale
            sin_scale = (i + 1.0) / self._total_num * 0.7
            sin_scale = -4 * (sin_scale - 0.5) ** 2 + 1
            # scale = -(self._times[i] - (xmin - xmax) / 2) + max_scale
            self._inputs.append(
                np.sin(self._times[i] * sin_scale) * (offset_scale - scale)
            )
        self._inputs = np.array(self._inputs)
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
        value = float(self._inputs[index])
        if self._transform is not None:
            value = self._transform(value)
        return index, float(self._times[index]), value

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements)".format(
            name=self.__class__.__name__, cur_num=self._total_num, total=len(self)
        )
