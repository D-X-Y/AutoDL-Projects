import paddle
import paddle.fluid as fluid
from .operations import OPS


def AuxiliaryHeadCIFAR(inputs, C, class_num):
  print ('AuxiliaryHeadCIFAR : inputs-shape : {:}'.format(inputs.shape))
  temp = fluid.layers.relu(inputs)
  temp = fluid.layers.pool2d(temp, pool_size=5, pool_stride=3, pool_padding=0, pool_type='avg')
  temp = fluid.layers.conv2d(temp, filter_size=1, num_filters=128, stride=1, padding=0, act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act='relu', bias_attr=None)
  temp = fluid.layers.conv2d(temp, filter_size=1, num_filters=768, stride=2, padding=0, act=None, bias_attr=False)
  temp = fluid.layers.batch_norm(input=temp, act='relu', bias_attr=None)
  print ('AuxiliaryHeadCIFAR : last---shape : {:}'.format(temp.shape))
  predict = fluid.layers.fc(input=temp, size=class_num, act='softmax')
  return predict


def InferCell(name, inputs_prev_prev, inputs_prev, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
  print ('[{:}] C_prev_prev={:} C_prev={:}, C={:}, reduction_prev={:}, reduction={:}'.format(name, C_prev_prev, C_prev, C, reduction_prev, reduction))
  print ('inputs_prev_prev : {:}'.format(inputs_prev_prev.shape))
  print ('inputs_prev      : {:}'.format(inputs_prev.shape))
  inputs_prev_prev = OPS['skip_connect'](inputs_prev_prev, C_prev_prev, C, 2 if reduction_prev else 1)
  inputs_prev      = OPS['skip_connect'](inputs_prev, C_prev, C, 1)
  print ('inputs_prev_prev : {:}'.format(inputs_prev_prev.shape))
  print ('inputs_prev      : {:}'.format(inputs_prev.shape))
  if reduction: step_ops, concat = genotype.reduce, genotype.reduce_concat
  else        : step_ops, concat = genotype.normal, genotype.normal_concat
  states = [inputs_prev_prev, inputs_prev]
  for istep, operations in enumerate(step_ops):
    op_a, op_b = operations
    # the first operation
    #print ('-->>[{:}/{:}] [{:}] + [{:}]'.format(istep, len(step_ops), op_a, op_b))
    stride  = 2 if reduction and op_a[1] < 2 else 1
    tensor1 = OPS[ op_a[0] ](states[op_a[1]], C, C, stride)
    stride  = 2 if reduction and op_b[1] < 2 else 1
    tensor2 = OPS[ op_b[0] ](states[op_b[1]], C, C, stride)
    state   = fluid.layers.elementwise_add(x=tensor1, y=tensor2, act=None)
    assert tensor1.shape == tensor2.shape, 'invalid shape {:} vs. {:}'.format(tensor1.shape, tensor2.shape)
    print ('-->>[{:}/{:}] tensor={:} from {:} + {:}'.format(istep, len(step_ops), state.shape, tensor1.shape, tensor2.shape))
    states.append( state )
  states_to_cat = [states[x] for x in concat]
  outputs = fluid.layers.concat(states_to_cat, axis=1)
  print ('-->> output-shape : {:} from concat={:}'.format(outputs.shape, concat))
  return outputs



# NASCifarNet(inputs, 36, 6, 3, 10, 'xxx', True)
def NASCifarNet(ipt, C, N, stem_multiplier, class_num, genotype, auxiliary):
  # cifar head module
  C_curr = stem_multiplier * C
  stem   = fluid.layers.conv2d(ipt, filter_size=3, num_filters=C_curr, stride=1, padding=1, act=None, bias_attr=False)
  stem   = fluid.layers.batch_norm(input=stem, act=None, bias_attr=None)
  print ('stem-shape : {:}'.format(stem.shape))
  # N + 1 + N + 1 + N cells
  layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
  layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
  C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
  reduction_prev = False
  auxiliary_pred = None

  cell_results = [stem, stem]
  for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
    xstr = '{:02d}/{:02d}'.format(index, len(layer_channels))
    cell_result    = InferCell(xstr, cell_results[-2], cell_results[-1], genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
    reduction_prev = reduction
    C_prev_prev, C_prev = C_prev, cell_result.shape[1]
    cell_results.append( cell_result )
    if auxiliary and reduction and C_curr == C*4:
      auxiliary_pred = AuxiliaryHeadCIFAR(cell_result, C_prev, class_num)

  global_P = fluid.layers.pool2d(input=cell_results[-1], pool_size=8, pool_type='avg', pool_stride=1)
  predicts = fluid.layers.fc(input=global_P, size=class_num, act='softmax')
  print ('predict-shape : {:}'.format(predicts.shape))
  if auxiliary_pred is None:
    return predicts
  else:
    return [predicts, auxiliary_pred]
