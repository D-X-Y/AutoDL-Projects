#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import math
import random
import numpy as np
from typing import List, Optional, Dict
import torch
import torch.utils.data as data

from .synthetic_utils import TimeStamp


def is_list_tuple(x):
    return isinstance(x, (tuple, list))


def zip_sequence(sequence):
    def _combine(*alist):
        if is_list_tuple(alist[0]):
            return [_combine(*xlist) for xlist in zip(*alist)]
        else:
            return torch.cat(alist, dim=0)

    def unsqueeze(a):
        if is_list_tuple(a):
            return [unsqueeze(x) for x in a]
        else:
            return a.unsqueeze(dim=0)

    with torch.no_grad():
        sequence = [unsqueeze(a) for a in sequence]
        return _combine(*sequence)


class SyntheticDEnv(data.Dataset):
    """The synethtic dynamic environment."""

    def __init__(
        self,
        mean_functors: List[data.Dataset],
        cov_functors: List[List[data.Dataset]],
        num_per_task: int = 5000,
        timestamp_config: Optional[Dict] = None,
        mode: Optional[str] = None,
        timestamp_noise_scale: float = 0.3,
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
        elif "mode" not in timestamp_config:
            timestamp_config["mode"] = mode

        self._timestamp_generator = TimeStamp(**timestamp_config)
        self._timestamp_noise_scale = timestamp_noise_scale

        self._mean_functors = mean_functors
        self._cov_functors = cov_functors

        self._oracle_map = None
        self._seq_length = None

    @property
    def min_timestamp(self):
        return self._timestamp_generator.min_timestamp

    @property
    def max_timestamp(self):
        return self._timestamp_generator.max_timestamp

    @property
    def timestamp_interval(self):
        return self._timestamp_generator.interval

    def random_timestamp(self):
        return (
            random.random() * (self.max_timestamp - self.min_timestamp)
            + self.min_timestamp
        )

    def reset_max_seq_length(self, seq_length):
        self._seq_length = seq_length

    def get_timestamp(self, index):
        if index is None:
            timestamps = []
            for index in range(len(self._timestamp_generator)):
                timestamps.append(self._timestamp_generator[index][1])
            return tuple(timestamps)
        else:
            index, timestamp = self._timestamp_generator[index]
            return timestamp

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
        if self._seq_length is None:
            return self.__call__(timestamp)
        else:
            noise = (
                random.random() * self.timestamp_interval * self._timestamp_noise_scale
            )
            timestamps = [
                timestamp + i * self.timestamp_interval + noise
                for i in range(self._seq_length)
            ]
            xdata = [self.__call__(timestamp) for timestamp in timestamps]
            return zip_sequence(xdata)

    def __call__(self, timestamp):
        mean_list = [functor(timestamp) for functor in self._mean_functors]
        cov_matrix = [
            [abs(cov_gen(timestamp)) for cov_gen in cov_functor]
            for cov_functor in self._cov_functors
        ]

        dataset = np.random.multivariate_normal(
            mean_list, cov_matrix, size=self._num_per_task
        )
        if self._oracle_map is None:
            return torch.Tensor([timestamp]), torch.Tensor(dataset)
        else:
            targets = self._oracle_map.noise_call(dataset, timestamp)
            return torch.Tensor([timestamp]), (
                torch.Tensor(dataset),
                torch.Tensor(targets),
            )

    def __len__(self):
        return len(self._timestamp_generator)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements, ndim={ndim}, num_per_task={num_per_task}, range=[{xrange_min:.5f}~{xrange_max:.5f}], mode={mode})".format(
            name=self.__class__.__name__,
            cur_num=len(self),
            total=len(self._timestamp_generator),
            ndim=self._ndim,
            num_per_task=self._num_per_task,
            xrange_min=self.min_timestamp,
            xrange_max=self.max_timestamp,
            mode=self._timestamp_generator.mode,
        )


class EnvSampler:
    def __init__(self, env, batch, enlarge):
        indexes = list(range(len(env)))
        self._indexes = indexes * enlarge
        self._batch = batch
        self._iterations = len(self._indexes) // self._batch

    def __iter__(self):
        random.shuffle(self._indexes)
        for it in range(self._iterations):
            indexes = self._indexes[it * self._batch : (it + 1) * self._batch]
            yield indexes

    def __len__(self):
        return self._iterations
