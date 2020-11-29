###############################################################
# NATS-Bench (https://arxiv.org/pdf/2009.00437.pdf)           #
# The code to draw some results in Table 4 in our paper.      #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-table.py                 #
###############################################################
import os, gc, sys, time, torch, argparse
import numpy as np
from typing import List, Text, Dict, Any
from shutil import copyfile
from collections import defaultdict, OrderedDict
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


def get_valid_test_acc(api, arch, dataset):
  is_size_space = api.search_space_name == 'size'
  if dataset == 'cifar10':
      xinfo = api.get_more_info(arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False)
      test_acc = xinfo['test-accuracy']
      xinfo = api.get_more_info(arch, dataset='cifar10-valid', hp=90 if is_size_space else 200, is_random=False)
      valid_acc = xinfo['valid-accuracy']
  else:
      xinfo = api.get_more_info(arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False)
      valid_acc = xinfo['valid-accuracy']
      test_acc = xinfo['test-accuracy']
  return valid_acc, test_acc, 'validation = {:.2f}, test = {:.2f}\n'.format(valid_acc, test_acc)


def show_valid_test(api, arch):
  is_size_space = api.search_space_name == 'size'
  final_str = ''
  for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
    valid_acc, test_acc, perf_str = get_valid_test_acc(api, arch, dataset)
    final_str += '{:} : {:}\n'.format(dataset, perf_str)
  return final_str


def find_best_valid(api, dataset):
  all_valid_accs, all_test_accs = [], []
  for index, arch in enumerate(api):
    # import pdb; pdb.set_trace()
    valid_acc, test_acc, perf_str = get_valid_test_acc(api, index, dataset)
    all_valid_accs.append((index, valid_acc))
    all_test_accs.append((index, test_acc))
  best_valid_index = sorted(all_valid_accs, key=lambda x: -x[1])[0][0]
  best_test_index = sorted(all_test_accs, key=lambda x: -x[1])[0][0]

  print('-' * 50 + '{:10s}'.format(dataset) + '-' * 50)
  print('Best ({:}) architecture on validation: {:}'.format(best_valid_index, api[best_valid_index]))
  print('Best ({:}) architecture on       test: {:}'.format(best_test_index, api[best_test_index]))
  _, _, perf_str = get_valid_test_acc(api, best_valid_index, dataset)
  print('using validation ::: {:}'.format(perf_str))
  _, _, perf_str = get_valid_test_acc(api, best_test_index, dataset)
  print('using test       ::: {:}'.format(perf_str))


if __name__ == '__main__':
  
  api_tss = create(None, 'tss', fast_mode=False, verbose=False)
  resnet = '|nor_conv_3x3~0|+|none~0|nor_conv_3x3~1|+|skip_connect~0|none~1|skip_connect~2|'
  resnet_index = api_tss.query_index_by_arch(resnet)
  print(show_valid_test(api_tss, resnet_index))

  for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
    find_best_valid(api_tss, dataset)

  largest = '64:64:64:64:64'
  largest_index = api_sss.query_index_by_arch(largest)
  print(show_valid_test(api_sss, largest_index))
  for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
    find_best_valid(api_sss, dataset)