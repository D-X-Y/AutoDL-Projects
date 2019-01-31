import torch
import random
import numpy as np

class ApplyOffset(object):
  def __init__(self, offset):
    assert isinstance(offset, int), 'The offset is not right : {}'.format(offset)
    self.offset = offset
  def __call__(self, x):
    if isinstance(x, np.ndarray) and x.dtype == 'uint8':
      x = x.astype(int)
    if isinstance(x, np.ndarray) and x.size == 1:
      x = int(x)
    return x + self.offset
