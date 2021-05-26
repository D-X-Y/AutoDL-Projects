import math
import abc
import numpy as np
from typing import Optional
import torch
import torch.utils.data as data


class UnifiedSplit:
    """A class to unify the split strategy."""

    def __init__(self, total_num, mode):
        # Training Set 75%
        num_of_train = int(total_num * 0.75)
        # Validation Set 05%
        num_of_valid = int(total_num * 0.05)
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
        elif mode.lower() in ("trainval", "trainvalid", "trainvalidation"):
            self._indexes = all_indexes[: num_of_train + num_of_valid]
        else:
            raise ValueError("Unkonwn mode of {:}".format(mode))
        self._all_indexes = all_indexes
        self._mode = mode

    @property
    def mode(self):
        return self._mode


class TimeStamp(UnifiedSplit, data.Dataset):
    """The timestamp dataset."""

    def __init__(
        self,
        min_timestamp: float = 0.0,
        max_timestamp: float = 1.0,
        num: int = 100,
        mode: Optional[str] = None,
    ):
        self._min_timestamp = min_timestamp
        self._max_timestamp = max_timestamp
        self._interval = (max_timestamp - min_timestamp) / (float(num) - 1)
        self._total_num = num
        UnifiedSplit.__init__(self, self._total_num, mode)

    @property
    def min_timestamp(self):
        return self._min_timestamp + self._interval * min(self._indexes)

    @property
    def max_timestamp(self):
        return self._min_timestamp + self._interval * max(self._indexes)

    @property
    def interval(self):
        return self._interval

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
        timestamp = self._min_timestamp + self._interval * index
        return index, timestamp

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements)".format(
            name=self.__class__.__name__,
            cur_num=len(self),
            total=self._total_num,
        )
