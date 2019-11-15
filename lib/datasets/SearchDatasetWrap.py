##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch, copy, random
import torch.utils.data as data


class SearchDataset(data.Dataset):

  def __init__(self, name, data, train_split, valid_split, check=True):
    self.datasetname = name
    self.data        = data
    self.train_split = train_split.copy()
    self.valid_split = valid_split.copy()
    if check:
      intersection = set(train_split).intersection(set(valid_split))
      assert len(intersection) == 0, 'the splitted train and validation sets should have no intersection'
    self.length      = len(self.train_split)

  def __repr__(self):
    return ('{name}(name={datasetname}, train={tr_L}, valid={val_L})'.format(name=self.__class__.__name__, datasetname=self.datasetname, tr_L=len(self.train_split), val_L=len(self.valid_split)))

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
    train_index = self.train_split[index]
    valid_index = random.choice( self.valid_split )
    train_image, train_label = self.data[train_index]
    valid_image, valid_label = self.data[valid_index]
    return train_image, train_label, valid_image, valid_label
