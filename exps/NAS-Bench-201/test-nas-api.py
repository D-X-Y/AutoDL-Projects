###############################################################
# NAS-Bench-201, ICLR 2020 (https://arxiv.org/abs/2001.00326) #
###############################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NAS-Bench-201/test-nas-api.py            #
###############################################################
import os, sys, time, torch, argparse
import numpy as np
from typing import List, Text, Dict, Any
from shutil import copyfile
from collections import defaultdict
from copy    import deepcopy
from pathlib import Path
import matplotlib
import seaborn as sns
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import dict2config, load_config
from nats_bench import create
from log_utils import time_string
from models import get_cell_based_tiny_net, CellStructure


def test_api(api, is_301=True):
  print('{:} start testing the api : {:}'.format(time_string(), api))
  api.clear_params(12)
  api.reload(index=12)
  
  # Query the informations of 1113-th architecture
  info_strs = api.query_info_str_by_arch(1113)
  print(info_strs)
  info = api.query_by_index(113)
  print('{:}\n'.format(info))
  info = api.query_by_index(113, 'cifar100')
  print('{:}\n'.format(info))

  info = api.query_meta_info_by_index(115, '90' if is_301 else '200')
  print('{:}\n'.format(info))

  for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
    for xset in ['train', 'test', 'valid']:
      best_index, highest_accuracy = api.find_best(dataset, xset)
    print('')
  params = api.get_net_param(12, 'cifar10', None)

  # Obtain the config and create the network
  config = api.get_net_config(12, 'cifar10')
  print('{:}\n'.format(config))
  network = get_cell_based_tiny_net(config)
  network.load_state_dict(next(iter(params.values())))

  # Obtain the cost information
  info = api.get_cost_info(12, 'cifar10')
  print('{:}\n'.format(info))
  info = api.get_latency(12, 'cifar10')
  print('{:}\n'.format(info))

  # Count the number of architectures
  info = api.statistics('cifar100', '12')
  print('{:}\n'.format(info))

  # Show the information of the 123-th architecture
  api.show(123)

  # Obtain both cost and performance information
  info = api.get_more_info(1234, 'cifar10')
  print('{:}\n'.format(info))
  print('{:} finish testing the api : {:}'.format(time_string(), api))

  if not is_301:
    arch_str = '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
    matrix = api.str2matrix(arch_str)
    print('Compute the adjacency matrix of {:}'.format(arch_str))
    print(matrix)
  info = api.simulate_train_eval(123, 'cifar10')
  print('simulate_train_eval : {:}'.format(info))


def test_issue_81_82(api):
  results = api.query_by_index(0, 'cifar10-valid', hp='12')
  results = api.query_by_index(0, 'cifar10-valid', hp='200')
  print(list(results.keys()))
  print(results[888].get_eval('valid'))
  print(results[888].get_eval('x-valid'))
  result_dict = api.get_more_info(index=0, dataset='cifar10-valid', iepoch=11, hp='200', is_random=False)
  info = api.query_by_arch('|nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|none~1|nor_conv_3x3~2|', '200')
  print(info)
  structure = CellStructure.str2structure('|nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|none~1|nor_conv_3x3~2|')
  info = api.query_by_arch(structure, '200')
  print(info)


if __name__ == '__main__':

  api201 = create(os.path.join(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_0-e61699.pth'), 'topology', True)
  test_issue_81_82(api201)
  print ('Test {:} done'.format(api201))

  api201 = create(None, 'topology', True)  # use the default file path
  test_issue_81_82(api201)
  test_api(api201, False)
  print ('Test {:} done'.format(api201))

  api301 = create(None, 'size', True)
  test_api(api301, True)
