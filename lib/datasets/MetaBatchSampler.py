# coding=utf-8
import numpy as np
import torch


class MetaBatchSampler(object):

  def __init__(self, labels, classes_per_it, num_samples, iterations):
    '''
    Initialize MetaBatchSampler
    Args:
    - labels: an iterable containing all the labels for the current dataset
    samples indexes will be infered from this iterable.
    - classes_per_it: number of random classes for each iteration
    - num_samples: number of samples for each iteration for each class (support + query)
    - iterations: number of iterations (episodes) per epoch
    '''
    super(MetaBatchSampler, self).__init__()
    self.labels           = labels.copy()
    self.classes_per_it   = classes_per_it
    self.sample_per_class = num_samples
    self.iterations       = iterations

    self.classes, self.counts = np.unique(self.labels, return_counts=True)
    assert len(self.classes) == np.max(self.classes) + 1 and np.min(self.classes) == 0
    assert classes_per_it < len(self.classes), '{:} vs. {:}'.format(classes_per_it, len(self.classes))
    self.classes = torch.LongTensor(self.classes)

    # create a matrix, indexes, of dim: classes X max(elements per class)
    # fill it with nans
    # for every class c, fill the relative row with the indices samples belonging to c
    # in numel_per_class we store the number of samples for each class/row
    self.indexes = { x.item() : [] for x in self.classes }
    indexes = { x.item() : [] for x in self.classes }

    for idx, label in enumerate(self.labels):
      indexes[ label.item() ].append( idx )
    for key, value in indexes.items():
      self.indexes[ key ] = torch.LongTensor( value )


  def __iter__(self):
    # yield a batch of indexes
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      batch = torch.LongTensor(batch_size)
      assert cpi < len(self.classes), '{:} vs. {:}'.format(cpi, len(self.classes))
      c_idxs = torch.randperm(len(self.classes))[:cpi]

      for i, cls in enumerate(self.classes[c_idxs]):
        s = slice(i * spc, (i + 1) * spc)
        num = self.indexes[ cls.item() ].nelement()
        assert spc < num, '{:} vs. {:}'.format(spc, num)
        sample_idxs = torch.randperm( num )[:spc]
        batch[s] = self.indexes[ cls.item() ][sample_idxs]

      batch = batch[torch.randperm(len(batch))]
      yield batch

  def __len__(self):
    # returns the number of iterations (episodes) per epoch
    return self.iterations
