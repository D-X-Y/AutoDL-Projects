import torch
import torch.nn as nn
# Squeeze and Excitation module

class SqEx(nn.Module):

  def __init__(self, n_features, reduction=16):
    super(SqEx, self).__init__()

    if n_features % reduction != 0:
      raise ValueError('n_features must be divisible by reduction (default = 16)')

    self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
    self.nonlin1 = nn.ReLU(inplace=True)
    self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
    self.nonlin2 = nn.Sigmoid()

  def forward(self, x):

    y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
    y = y.permute(0, 2, 3, 1)
    y = self.nonlin1(self.linear1(y))
    y = self.nonlin2(self.linear2(y))
    y = y.permute(0, 3, 1, 2)
    y = x * y
    return y

