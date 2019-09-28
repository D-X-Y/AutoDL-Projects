import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import OPS, FactorizedReduce, ReLUConvBN, Identity


def random_select(length, ratio):
  clist = []
  index = random.randint(0, length-1)
  for i in range(length):
    if i == index or random.random() < ratio:
      clist.append( 1 )
    else:
      clist.append( 0 )
  return clist


def all_select(length):
  return [1 for i in range(length)]


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    mask = x.new_zeros(x.size(0), 1, 1, 1)
    mask = mask.bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def return_alphas_str(basemodel):
  string = 'normal : {:}'.format( F.softmax(basemodel.alphas_normal, dim=-1) )
  if hasattr(basemodel, 'alphas_reduce'):
    string = string + '\nreduce : {:}'.format( F.softmax(basemodel.alphas_reduce, dim=-1) )
  return string


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices, values = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices, values = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, values, concat, reduction)

  def _compile(self, C, op_names, indices, values, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops.append( op )
    self._indices = indices
    self._values  = values

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)

      s = h1 + h2

      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)



class Transition(nn.Module):

  def __init__(self, C_prev_prev, C_prev, C, reduction_prev, multiplier=4):
    super(Transition, self).__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    self.multiplier  = multiplier

    self.reduction = True
    self.ops1 = nn.ModuleList(
                  [nn.Sequential(
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False),
                      nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False),
                      nn.BatchNorm2d(C, affine=True),
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
                      nn.BatchNorm2d(C, affine=True)),
                   nn.Sequential(
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False),
                      nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False),
                      nn.BatchNorm2d(C, affine=True),
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
                      nn.BatchNorm2d(C, affine=True))])

    self.ops2 = nn.ModuleList(
                  [nn.Sequential(
                      nn.MaxPool2d(3, stride=2, padding=1),
                      nn.BatchNorm2d(C, affine=True)),
                   nn.Sequential(
                      nn.MaxPool2d(3, stride=2, padding=1),
                      nn.BatchNorm2d(C, affine=True))])


  def forward(self, s0, s1, drop_prob = -1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    X0 = self.ops1[0] (s0)
    X1 = self.ops1[1] (s1)
    if self.training and drop_prob > 0.:
      X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

    #X2 = self.ops2[0] (X0+X1)
    X2 = self.ops2[0] (s0)
    X3 = self.ops2[1] (s1)
    if self.training and drop_prob > 0.:
      X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
    return torch.cat([X0, X1, X2, X3], dim=1)
