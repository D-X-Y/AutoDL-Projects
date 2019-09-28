import torch, copy, random
import torch.utils.data as data


class SearchDataset(data.Dataset):

  def __init__(self, name, data, train_split, valid_split):
    self.datasetname = name
    self.data        = data
    self.train_split = train_split.copy()
    self.valid_split = valid_split.copy()
    self.length      = len(self.train_split)

  def __repr__(self):
    return ('{name}(name={datasetname}, length={length})'.format(name=self.__class__.__name__, **self.__dict__))

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
    train_index = self.train_split[index]
    valid_index = random.choice( self.valid_split )
    train_image, train_label = self.data[train_index]
    valid_image, valid_label = self.data[valid_index]
    return train_image, train_label, valid_image, valid_label
