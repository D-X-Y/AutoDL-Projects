#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import math
import abc
import numpy as np
from typing import List, Optional, Dict
import torch
import torch.utils.data as data

from .synthetic_utils import TimeStamp


class SyntheticDEnv(data.Dataset):
    """The synethtic dynamic environment."""

    def __init__(
        self,
        mean_functors: List[data.Dataset],
        cov_functors: List[List[data.Dataset]],
        num_per_task: int = 5000,
        timestamp_config: Optional[Dict] = None,
        mode: Optional[str] = None,
    ):
        self._ndim = len(mean_functors)
        assert self._ndim == len(
            cov_functors
        ), "length does not match {:} vs. {:}".format(self._ndim, len(cov_functors))
        for cov_functor in cov_functors:
            assert self._ndim == len(
                cov_functor
            ), "length does not match {:} vs. {:}".format(self._ndim, len(cov_functor))
        self._num_per_task = num_per_task
        if timestamp_config is None:
            timestamp_config = dict(mode=mode)
        else:
            timestamp_config["mode"] = mode

        self._timestamp_generator = TimeStamp(**timestamp_config)

        self._mean_functors = mean_functors
        self._cov_functors = cov_functors

        self._oracle_map = None

    @property
    def min_timestamp(self):
        return self._timestamp_generator.min_timestamp

    @property
    def max_timestamp(self):
        return self._timestamp_generator.max_timestamp

    def set_oracle_map(self, functor):
        self._oracle_map = functor

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
        index, timestamp = self._timestamp_generator[index]
        mean_list = [functor(timestamp) for functor in self._mean_functors]
        cov_matrix = [
            [abs(cov_gen(timestamp)) for cov_gen in cov_functor]
            for cov_functor in self._cov_functors
        ]

        dataset = np.random.multivariate_normal(
            mean_list, cov_matrix, size=self._num_per_task
        )
        if self._oracle_map is None:
            return timestamp, torch.Tensor(dataset)
        else:
            targets = self._oracle_map.noise_call(dataset, timestamp)
            return timestamp, (torch.Tensor(dataset), torch.Tensor(targets))

    def __len__(self):
        return len(self._timestamp_generator)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements, ndim={ndim}, num_per_task={num_per_task})".format(
            name=self.__class__.__name__,
            cur_num=len(self),
            total=len(self._timestamp_generator),
            ndim=self._ndim,
            num_per_task=self._num_per_task,
        )
