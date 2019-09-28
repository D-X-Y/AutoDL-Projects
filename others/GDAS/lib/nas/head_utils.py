import torch
import torch.nn as nn


class ImageNetHEAD(nn.Sequential):
  def __init__(self, C, stride=2):
    super(ImageNetHEAD, self).__init__()
    self.add_module('conv1', nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False))
    self.add_module('bn1'  , nn.BatchNorm2d(C // 2))
    self.add_module('relu1', nn.ReLU(inplace=True))
    self.add_module('conv2', nn.Conv2d(C // 2, C, kernel_size=3, stride=stride, padding=1, bias=False))
    self.add_module('bn2'  , nn.BatchNorm2d(C))


class CifarHEAD(nn.Sequential):
  def __init__(self, C):
    super(CifarHEAD, self).__init__()
    self.add_module('conv', nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False))
    self.add_module('bn', nn.BatchNorm2d(C))
