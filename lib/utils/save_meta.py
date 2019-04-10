##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import os, sys
import os.path as osp
import numpy as np

def tensor2np(x):
  if isinstance(x, np.ndarray): return x
  if x.is_cuda: x = x.cpu()
  return x.numpy()

class Save_Meta():

  def __init__(self):
    self.reset()

  def __repr__(self):
    return ('{name}'.format(name=self.__class__.__name__)+'(number of data = {})'.format(len(self)))

  def reset(self):
    self.predictions = []
    self.groundtruth = []
    
  def __len__(self):
    return len(self.predictions)

  def append(self, _pred, _ground):
    _pred, _ground = tensor2np(_pred), tensor2np(_ground)
    assert _ground.shape[0] == _pred.shape[0] and len(_pred.shape) == 2 and len(_ground.shape) == 1, 'The shapes are wrong : {} & {}'.format(_pred.shape, _ground.shape)
    self.predictions.append(_pred)
    self.groundtruth.append(_ground)

  def save(self, save_dir, filename, test=True):
    meta = {'predictions': self.predictions, 
            'groundtruth': self.groundtruth}
    filename = osp.join(save_dir, filename)
    torch.save(meta, filename)
    if test:
      predictions = np.concatenate(self.predictions)
      groundtruth = np.concatenate(self.groundtruth)
      predictions = np.argmax(predictions, axis=1)
      accuracy = np.sum(groundtruth==predictions) * 100.0 / predictions.size
    else:
      accuracy = None
    print ('save save_meta into {} with accuracy = {}'.format(filename, accuracy))

  def load(self, filename):
    assert os.path.isfile(filename), '{} is not a file'.format(filename)
    checkpoint       = torch.load(filename)
    self.predictions = checkpoint['predictions']
    self.groundtruth = checkpoint['groundtruth']
