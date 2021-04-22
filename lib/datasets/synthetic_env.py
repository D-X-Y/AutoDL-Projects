#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import math
import abc
import numpy as np
from typing import List, Optional
import torch
import torch.utils.data as data

from .synthetic_utils import UnifiedSplit


class SyntheticDEnv(UnifiedSplit, data.Dataset):
    """The synethtic dynamic environment."""

    def __init__(
        self,
        mean_generators: List[data.Dataset],
        cov_generators: List[List[data.Dataset]],
        num_per_task: int = 5000,
        mode: Optional[str] = None,
    ):
        self._ndim = len(mean_generators)
        assert self._ndim == len(
            cov_generators
        ), "length does not match {:} vs. {:}".format(self._ndim, len(cov_generators))
        for cov_generator in cov_generators:
            assert self._ndim == len(
                cov_generator
            ), "length does not match {:} vs. {:}".format(
                self._ndim, len(cov_generator)
            )
        self._num_per_task = num_per_task
        self._total_num = len(mean_generators[0])
        for mean_generator in mean_generators:
            assert self._total_num == len(mean_generator)
        for cov_generator in cov_generators:
            for cov_g in cov_generator:
                assert self._total_num == len(cov_g)

        self._mean_generators = mean_generators
        self._cov_generators = cov_generators

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
        mean_list = [generator[index][-1] for generator in self._mean_generators]
        cov_matrix = [
            [cov_gen[index][-1] for cov_gen in cov_generator]
            for cov_generator in self._cov_generators
        ]

        dataset = np.random.multivariate_normal(
            mean_list, cov_matrix, size=self._num_per_task
        )
        return index, torch.Tensor(dataset)

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements, ndim={ndim}, num_per_task={num_per_task})".format(
            name=self.__class__.__name__,
            cur_num=len(self),
            total=self._total_num,
            ndim=self._ndim,
            num_per_task=self._num_per_task,
        )
