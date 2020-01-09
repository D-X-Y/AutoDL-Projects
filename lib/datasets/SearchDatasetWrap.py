##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch, copy, random
import torch.utils.data as data


class SearchDataset(data.Dataset):

  def __init__(self, name, data, train_split, valid_split, check=True):
    self.datasetname = name
    if isinstance(data, (list, tuple)): # new type of SearchDataset
      assert len(data) == 2, 'invalid length: {:}'.format( len(data) )
      self.train_data  = data[0]
      self.valid_data  = data[1]
      self.train_split = train_split.copy()
      self.valid_split = valid_split.copy()
      self.mode_str    = 'V2' # new mode 
    else:
      self.mode_str    = 'V1' # old mode 
      self.data        = data
      self.train_split = train_split.copy()
      self.valid_split = valid_split.copy()
      if check:
        intersection = set(train_split).intersection(set(valid_split))
        assert len(intersection) == 0, 'the splitted train and validation sets should have no intersection'
    self.length      = len(self.train_split)

  def __repr__(self):
    return ('{name}(name={datasetname}, train={tr_L}, valid={val_L}, version={ver})'.format(name=self.__class__.__name__, datasetname=self.datasetname, tr_L=len(self.train_split), val_L=len(self.valid_split), ver=self.mode_str))

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
    train_index = self.train_split[index]
    valid_index = random.choice( self.valid_split )
    if self.mode_str == 'V1':
      train_image, train_label = self.data[train_index]
      valid_image, valid_label = self.data[valid_index]
    elif self.mode_str == 'V2':
      train_image, train_label = self.train_data[train_index]
      valid_image, valid_label = self.valid_data[valid_index]
    else: raise ValueError('invalid mode : {:}'.format(self.mode_str))
    return train_image, train_label, valid_image, valid_label
