import torch
import torch.nn as nn

__all__ = ['OPS', 'ReLUConvBN', 'SearchSpaceNames']

OPS = {
  'none'         : lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
  'avg_pool_3x3' : lambda C_in, C_out, stride: POOLING(C_in, C_out, stride, 'avg'),
  'max_pool_3x3' : lambda C_in, C_out, stride: POOLING(C_in, C_out, stride, 'max'),
  'nor_conv_7x7' : lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (7,7), (stride,stride), (3,3), (1,1)),
  'nor_conv_3x3' : lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1)),
  'nor_conv_1x1' : lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1)),
  'skip_connect' : lambda C_in, C_out, stride: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride),
}

CONNECT_NAS_BENCHMARK  = ['none', 'skip_connect', 'nor_conv_3x3']

SearchSpaceNames = {'connect-nas' : CONNECT_NAS_BENCHMARK}


class POOLING(nn.Module):

  def __init__(self, C_in, C_out, stride, mode):
    super(POOLING, self).__init__()
    if C_in == C_out:
      self.preprocess = None
    else:
      self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0)
    if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
    else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

  def forward(self, inputs):
    if self.preprocess: x = self.preprocess(inputs)
    else              : x = inputs
    return self.op(x)


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out)
    )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.C_in   = C_in
    self.C_out  = C_out
    self.stride = stride
    self.is_zero = True

  def forward(self, x):
    if self.C_in == self.C_out:
      if self.stride == 1: return x.mul(0.)
      else               : return x[:,:,::self.stride,::self.stride].mul(0.)
    else:
      shape = list(x.shape)
      shape[1] = self.C_out
      zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
      return zeros

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, stride):
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
    else:
      raise ValueError('Invalid stride : {:}'.format(stride))
    
    self.bn = nn.BatchNorm2d(C_out)

  def forward(self, x):
    x = self.relu(x)
    y = self.pad(x)
    out = torch.cat([self.convs[0](x), self.convs[1](y[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)
