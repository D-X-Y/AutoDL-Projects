##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# modified from https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/utils/flops_benchmark.py
import copy, torch

def print_FLOPs(model, shape, logs):
  print_log, log = logs
  model = copy.deepcopy( model )

  model = add_flops_counting_methods(model)
  model = model.cuda()
  model.eval()

  cache_inputs = torch.zeros(*shape).cuda()
  #print_log('In the calculating function : cache input size : {:}'.format(cache_inputs.size()), log)
  _ = model(cache_inputs)
  FLOPs = compute_average_flops_cost( model ) / 1e6
  print_log('FLOPs : {:} MB'.format(FLOPs), log)
  torch.cuda.empty_cache()


# ---- Public functions
def add_flops_counting_methods( model ):
  model.__batch_counter__ = 0
  add_batch_counter_hook_function( model )
  model.apply( add_flops_counter_variable_or_reset )
  model.apply( add_flops_counter_hook_function )
  return model



def compute_average_flops_cost(model):
  """
  A method that will be available after add_flops_counting_methods() is called on a desired net object.
  Returns current mean flops consumption per image.
  """
  batches_count = model.__batch_counter__
  flops_sum = 0
  for module in model.modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      flops_sum += module.__flops__
  return flops_sum / batches_count


# ---- Internal functions
def pool_flops_counter_hook(pool_module, inputs, output):
  batch_size = inputs[0].size(0)
  kernel_size = pool_module.kernel_size
  out_C, output_height, output_width = output.shape[1:]
  assert out_C == inputs[0].size(1), '{:} vs. {:}'.format(out_C, inputs[0].size())

  overall_flops = batch_size * out_C * output_height * output_width * kernel_size * kernel_size
  pool_module.__flops__ += overall_flops


def fc_flops_counter_hook(fc_module, inputs, output):
  batch_size = inputs[0].size(0)
  xin, xout = fc_module.in_features, fc_module.out_features
  assert xin == inputs[0].size(1) and xout == output.size(1), 'IO=({:}, {:})'.format(xin, xout)
  overall_flops = batch_size * xin * xout
  if fc_module.bias is not None:
    overall_flops += batch_size * xout
  fc_module.__flops__ += overall_flops


def conv_flops_counter_hook(conv_module, inputs, output):
  batch_size = inputs[0].size(0)
  output_height, output_width = output.shape[2:]
  
  kernel_height, kernel_width = conv_module.kernel_size
  in_channels  = conv_module.in_channels
  out_channels = conv_module.out_channels
  groups       = conv_module.groups
  conv_per_position_flops = kernel_height * kernel_width * in_channels * out_channels / groups
  
  active_elements_count = batch_size * output_height * output_width
  overall_flops = conv_per_position_flops * active_elements_count
    
  if conv_module.bias is not None:
    overall_flops += out_channels * active_elements_count
  conv_module.__flops__ += overall_flops

  
def batch_counter_hook(module, inputs, output):
  # Can have multiple inputs, getting the first one
  inputs = inputs[0]
  batch_size = inputs.shape[0]
  module.__batch_counter__ += batch_size


def add_batch_counter_hook_function(module):
  if not hasattr(module, '__batch_counter_handle__'):
    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle

  
def add_flops_counter_variable_or_reset(module):
  if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) \
    or isinstance(module, torch.nn.AvgPool2d) or isinstance(module, torch.nn.MaxPool2d):
    module.__flops__ = 0


def add_flops_counter_hook_function(module):
  if isinstance(module, torch.nn.Conv2d):
    if not hasattr(module, '__flops_handle__'):
      handle = module.register_forward_hook(conv_flops_counter_hook)
      module.__flops_handle__ = handle
  elif isinstance(module, torch.nn.Linear):
    if not hasattr(module, '__flops_handle__'):
      handle = module.register_forward_hook(fc_flops_counter_hook)
      module.__flops_handle__ = handle
  elif isinstance(module, torch.nn.AvgPool2d) or isinstance(module, torch.nn.MaxPool2d):
    if not hasattr(module, '__flops_handle__'):
      handle = module.register_forward_hook(pool_flops_counter_hook)
      module.__flops_handle__ = handle
