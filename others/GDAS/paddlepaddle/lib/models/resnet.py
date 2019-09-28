import paddle
import paddle.fluid as fluid


def conv_bn_layer(input,
          ch_out,
          filter_size,
          stride,
          padding,
          act='relu',
          bias_attr=False):
  tmp = fluid.layers.conv2d(
    input=input,
    filter_size=filter_size,
    num_filters=ch_out,
    stride=stride,
    padding=padding,
    act=None,
    bias_attr=bias_attr)
  return fluid.layers.batch_norm(input=tmp, act=act)


def shortcut(input, ch_in, ch_out, stride):
  if stride == 2:
    temp = fluid.layers.pool2d(input, pool_size=2, pool_type='avg', pool_stride=2)
    temp = fluid.layers.conv2d(temp , filter_size=1, num_filters=ch_out, stride=1, padding=0, act=None, bias_attr=None)
    return temp
  elif ch_in != ch_out:
    return conv_bn_layer(input, ch_out, 1, stride, 0, None, None)
  else:
    return input


def basicblock(input, ch_in, ch_out, stride):
  tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
  tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
  short = shortcut(input, ch_in, ch_out, stride)
  return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride):
  tmp = block_func(input, ch_in, ch_out, stride)
  for i in range(1, count):
    tmp = block_func(tmp, ch_out, ch_out, 1)
  return tmp


def resnet_cifar(ipt, depth, class_num):
  # depth should be one of 20, 32, 44, 56, 110, 1202
  assert (depth - 2) % 6 == 0
  n = (depth - 2) // 6
  print('[resnet] depth : {:}, class_num : {:}'.format(depth, class_num))
  conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
  print('conv-1 : shape = {:}'.format(conv1.shape))
  res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
  print('res--1 : shape = {:}'.format(res1.shape))
  res2 = layer_warp(basicblock, res1 , 16, 32, n, 2)
  print('res--2 : shape = {:}'.format(res2.shape))
  res3 = layer_warp(basicblock, res2 , 32, 64, n, 2)
  print('res--3 : shape = {:}'.format(res3.shape))
  pool = fluid.layers.pool2d(input=res3, pool_size=8, pool_type='avg', pool_stride=1)
  print('pool   : shape = {:}'.format(pool.shape))
  predict = fluid.layers.fc(input=pool, size=class_num, act='softmax')
  print('predict: shape = {:}'.format(predict.shape))
  return predict
