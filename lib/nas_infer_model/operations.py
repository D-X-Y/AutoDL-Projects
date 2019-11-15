##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn

OPS = {
  'none'         : lambda C_in, C_out, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C_in, C_out, stride, affine: POOLING(C_in, C_out, stride, 'avg'),
  'max_pool_3x3' : lambda C_in, C_out, stride, affine: POOLING(C_in, C_out, stride, 'max'),
  'nor_conv_7x7' : lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, (7,7), (stride,stride), (3,3), affine),
  'nor_conv_3x3' : lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), affine),
  'nor_conv_1x1' : lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), affine),
  'skip_connect' : lambda C_in, C_out, stride, affine: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine),
  'sep_conv_3x3' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C_in, C_out, stride, affine: Conv717(C_in, C_out, stride, affine),
  'conv_3x1_1x3' : lambda C_in, C_out, stride, affine: Conv313(C_in, C_out, stride, affine)
}


class POOLING(nn.Module):

  def __init__(self, C_in, C_out, stride, mode):
    super(POOLING, self).__init__()
    if C_in == C_out:
      self.preprocess = None
    else:
      self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0)
    if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)

  def forward(self, inputs):
    if self.preprocess is not None:
      x = self.preprocess(inputs)
    else: x = inputs
    return self.op(x)


class Conv313(nn.Module):

  def __init__(self, C_in, C_out, stride, affine):
    super(Conv313, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in , C_out, (1,3), stride=(1, stride), padding=(0, 1), bias=False),
      nn.Conv2d(C_out, C_out, (3,1), stride=(stride, 1), padding=(1, 0), bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class Conv717(nn.Module):

  def __init__(self, C_in, C_out, stride, affine):
    super(Conv717, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in , C_out, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
      nn.Conv2d(C_out, C_out, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in,  kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=     1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

  def extra_repr(self):
    return 'stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, stride, affine=True):
    super(FactorizedReduce, self).__init__()
    self.stride = stride
    self.C_in   = C_in  
    self.C_out  = C_out  
    self.relu   = nn.ReLU(inplace=False)
    if stride == 2:
      #assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
      C_outs = [C_out // 2, C_out - C_out // 2]
      self.convs = nn.ModuleList()
      for i in range(2):
        self.convs.append( nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False) )
      self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
    elif stride == 4:
      assert C_out % 4 == 0, 'C_out : {:}'.format(C_out)
      self.convs = nn.ModuleList()
      for i in range(4):
        self.convs.append( nn.Conv2d(C_in, C_out // 4, 1, stride=stride, padding=0, bias=False) )
      self.pad = nn.ConstantPad2d((0, 3, 0, 3), 0)
    else:
      raise ValueError('Invalid stride : {:}'.format(stride))
    
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    y = self.pad(x)
    if self.stride == 2:
      out = torch.cat([self.convs[0](x), self.convs[1](y[:,:,1:,1:])], dim=1)
    else:
      out = torch.cat([self.convs[0](x),            self.convs[1](y[:,:,1:-2,1:-2]),
                       self.convs[2](y[:,:,2:-1,2:-1]), self.convs[3](y[:,:,3:,3:])], dim=1)
    out = self.bn(out)
    return out

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)
