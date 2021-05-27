import numpy as np
import torch
import torch.utils.data as data


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
        data_generator,
        oracle_map,
        time_generator,
        num_per_task: int = 5000,
        noise: float = 0.1,
    ):
        self._data_generator = data_generator
        self._time_generator = time_generator
        self._oracle_map = oracle_map
        self._num_per_task = num_per_task
        self._noise = noise
        self._meta_info = dict()

    def set_regression(self):
        self._meta_info["task"] = "regression"
        self._meta_info["input_dim"] = self._data_generator.ndim
        self._meta_info["output_shape"] = self._oracle_map.output_shape(
            self._data_generator.output_shape()
        )
        self._meta_info["output_dim"] = int(np.prod(self._meta_info["output_shape"]))

    def set_classification(self, num_classes):
        self._meta_info["task"] = "classification"
        self._meta_info["input_dim"] = self._data_generator.ndim
        self._meta_info["num_classes"] = int(num_classes)
        self._meta_info["output_shape"] = self._oracle_map.output_shape(
            self._data_generator.output_shape()
        )
        self._meta_info["output_dim"] = int(np.prod(self._meta_info["output_shape"]))

    @property
    def oracle_map(self):
        return self._oracle_map

    @property
    def meta_info(self):
        return self._meta_info

    @property
    def min_timestamp(self):
        return self._time_generator.min_timestamp

    @property
    def max_timestamp(self):
        return self._time_generator.max_timestamp

    @property
    def time_interval(self):
        return self._time_generator.interval

    @property
    def mode(self):
        return self._time_generator.mode

    def get_seq_times(self, index, seq_length):
        index, timestamp = self._time_generator[index]
        xtimes = []
        for i in range(1, seq_length + 1):
            xtimes.append(timestamp - i * self.time_interval)
        xtimes.reverse()
        return xtimes

    def get_timestamp(self, index):
        if index is None:
            timestamps = []
            for index in range(len(self._time_generator)):
                timestamps.append(self._time_generator[index][1])
            return tuple(timestamps)
        else:
            index, timestamp = self._time_generator[index]
            return timestamp

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
        index, timestamp = self._time_generator[index]
        return self.__call__(timestamp)

    def seq_call(self, timestamps):
        with torch.no_grad():
            if isinstance(timestamps, torch.Tensor):
                timestamps = timestamps.cpu().tolist()
            xdata = [self.__call__(timestamp) for timestamp in timestamps]
            return zip_sequence(xdata)

    def __call__(self, timestamp):
        dataset = self._data_generator(timestamp, self._num_per_task)
        targets = self._oracle_map.noise_call(dataset, timestamp, self._noise)
        if isinstance(dataset, np.ndarray):
            dataset = torch.from_numpy(dataset)
        else:
            dataset = torch.Tensor(dataset)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        else:
            targets = torch.Tensor(targets)
        if dataset.dtype == torch.float64:
            dataset = dataset.float()
        if targets.dtype == torch.float64:
            targets = targets.float()
        return torch.Tensor([timestamp]), (dataset, targets)

    def __len__(self):
        return len(self._time_generator)

    def __repr__(self):
        return "{name}({cur_num:}/{total} elements, ndim={ndim}, num_per_task={num_per_task}, range=[{xrange_min:.5f}~{xrange_max:.5f}], mode={mode})".format(
            name=self.__class__.__name__,
            cur_num=len(self),
            total=len(self._time_generator),
            ndim=self._data_generator.ndim,
            num_per_task=self._num_per_task,
            xrange_min=self.min_timestamp,
            xrange_max=self.max_timestamp,
            mode=self.mode,
        )
