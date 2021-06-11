#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.06 #
#####################################################
import random


class BatchSampler:
    """A batch sampler used for single machine training."""

    def __init__(self, dataset, batch, steps):
        self._num_per_epoch = len(dataset)
        self._iter_per_epoch = self._num_per_epoch // batch
        self._steps = steps
        self._batch = batch
        if self._num_per_epoch < self._batch:
            raise ValueError(
                "The dataset size must be larger than batch={:}".format(batch)
            )
        self._indexes = list(range(self._num_per_epoch))

    def __iter__(self):
        """
        yield a batch of indexes using random sampling
        """
        for i in range(self._steps):
            if i % self._iter_per_epoch == 0:
                random.shuffle(self._indexes)
            j = i % self._iter_per_epoch
            yield self._indexes[j * self._batch : (j + 1) * self._batch]

    def __len__(self):
        return self._steps
