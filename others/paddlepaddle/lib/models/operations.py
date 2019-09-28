import paddle
import paddle.fluid as fluid


OPS = {
  'none'         : lambda inputs, C_in, C_out, stride: ZERO(inputs, stride),
  'avg_pool_3x3' : lambda inputs, C_in, C_out, stride: POOL_3x3(inputs, C_in, C_out, stride, 'avg'),
  'max_pool_3x3' : lambda inputs, C_in, C_out, stride: POOL_3x3(inputs, C_in, C_out, stride, 'max'),
  'skip_connect' : lambda inputs, C_in, C_out, stride: Identity(inputs, C_in, C_out, stride),
  'sep_conv_3x3' : lambda inputs, C_in, C_out, stride: SepConv(inputs, C_in, C_out, 3, stride, 1),
  'sep_conv_5x5' : lambda inputs, C_in, C_out, stride: SepConv(inputs, C_in, C_out, 5, stride, 2),
  'sep_conv_7x7' : lambda inputs, C_in, C_out, stride: SepConv(inputs, C_in, C_out, 7, stride, 3),
  'dil_conv_3x3' : lambda inputs, C_in, C_out, stride: DilConv(inputs, C_in, C_out, 3, stride, 2, 2),
  'dil_conv_5x5' : lambda inputs, C_in, C_out, stride: DilConv(inputs, C_in, C_out, 5, stride, 4, 2),
  'conv_3x1_1x3' : lambda inputs, C_in, C_out, stride: Conv313(inputs, C_in, C_out, stride),
  'conv_7x1_1x7' : lambda inputs, C_in, C_out, stride: Conv717(inputs, C_in, C_out, stride),
}


def ReLUConvBN(inputs, C_in, C_out, kernel, stride, padding):
  temp = fluid.layers.relu(inputs)
  temp = fluid.layers.conv2d(temp, filter_size=kernel, num_filters=C_out, stride=stride, padding=padding, act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act=None, bias_attr=None)
  return temp


def ZERO(inputs, stride):
  if stride == 1:
    return inputs * 0
  elif stride == 2:
    return fluid.layers.pool2d(inputs, filter_size=2, pool_stride=2, pool_padding=0, pool_type='avg') * 0
  else:
    raise ValueError('invalid stride of {:} not [1, 2]'.format(stride))


def Identity(inputs, C_in, C_out, stride):
  if C_in == C_out and stride == 1:
    return inputs
  elif stride == 1:
    return ReLUConvBN(inputs, C_in, C_out, 1, 1, 0)
  else:
    temp1 = fluid.layers.relu(inputs)
    temp2 = fluid.layers.pad2d(input=temp1, paddings=[0, 1, 0, 1], mode='reflect')
    temp2 = fluid.layers.slice(temp2, axes=[0, 1, 2, 3], starts=[0, 0, 1, 1], ends=[999, 999, 999, 999])
    temp1 = fluid.layers.conv2d(temp1, filter_size=1, num_filters=C_out//2, stride=stride, padding=0, act=None, bias_attr=False)
    temp2 = fluid.layers.conv2d(temp2, filter_size=1, num_filters=C_out-C_out//2, stride=stride, padding=0, act=None, bias_attr=False)
    temp  = fluid.layers.concat([temp1,temp2], axis=1)
    return fluid.layers.batch_norm(input=temp, act=None, bias_attr=None)


def POOL_3x3(inputs, C_in, C_out, stride, mode):
  if C_in == C_out:
    xinputs = inputs
  else:
    xinputs = ReLUConvBN(inputs, C_in, C_out, 1, 1, 0)
  return fluid.layers.pool2d(xinputs, pool_size=3, pool_stride=stride, pool_padding=1, pool_type=mode)


def SepConv(inputs, C_in, C_out, kernel, stride, padding):
  temp = fluid.layers.relu(inputs)
  temp = fluid.layers.conv2d(temp, filter_size=kernel, num_filters=C_in , stride=stride, padding=padding, act=None, bias_attr=False)
  temp = fluid.layers.conv2d(temp, filter_size=     1, num_filters=C_in , stride=     1, padding=      0, act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act='relu', bias_attr=None)
  temp = fluid.layers.conv2d(temp, filter_size=kernel, num_filters=C_in , stride=     1, padding=padding, act=None, bias_attr=False)
  temp = fluid.layers.conv2d(temp, filter_size=     1, num_filters=C_out, stride=     1, padding=      0, act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act=None  , bias_attr=None)
  return temp


def DilConv(inputs, C_in, C_out, kernel, stride, padding, dilation):
  temp = fluid.layers.relu(inputs)
  temp = fluid.layers.conv2d(temp, filter_size=kernel, num_filters=C_in , stride=stride, padding=padding, dilation=dilation, act=None, bias_attr=False)
  temp = fluid.layers.conv2d(temp, filter_size=     1, num_filters=C_out, stride=     1, padding=      0, act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act=None, bias_attr=None)
  return temp


def Conv313(inputs, C_in, C_out, stride):
  temp = fluid.layers.relu(inputs)
  temp = fluid.layers.conv2d(temp, filter_size=(1,3), num_filters=C_out, stride=(1,stride), padding=(0,1), act=None, bias_attr=False)
  temp = fluid.layers.conv2d(temp, filter_size=(3,1), num_filters=C_out, stride=(stride,1), padding=(1,0), act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act=None, bias_attr=None)
  return temp


def Conv717(inputs, C_in, C_out, stride):
  temp = fluid.layers.relu(inputs)
  temp = fluid.layers.conv2d(temp, filter_size=(1,7), num_filters=C_out, stride=(1,stride), padding=(0,3), act=None, bias_attr=False)
  temp = fluid.layers.conv2d(temp, filter_size=(7,1), num_filters=C_out, stride=(stride,1), padding=(3,0), act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act=None, bias_attr=None)
  return temp
