###############################################################
# NAS-Bench-201, ICLR 2020 (https://arxiv.org/abs/2001.00326) #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NAS-Bench-201/test-nas-api.py
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
from nas_201_api import NASBench201API, NASBench301API
from log_utils import time_string
from models import get_cell_based_tiny_net


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

  # obtain the config and create the network
  config = api.get_net_config(12, 'cifar10')
  print('{:}\n'.format(config))
  network = get_cell_based_tiny_net(config)
  network.load_state_dict(next(iter(params.values())))

  # obtain the cost information
  info = api.get_cost_info(12, 'cifar10')
  print('{:}\n'.format(info))
  info = api.get_latency(12, 'cifar10')
  print('{:}\n'.format(info))

  # count the number of architectures
  info = api.statistics('cifar100', '12')
  print('{:}\n'.format(info))

  # show the information of the 123-th architecture
  api.show(123)

  # obtain both cost and performance information
  info = api.get_more_info(1234, 'cifar10')
  print('{:}\n'.format(info))
  print('{:} finish testing the api : {:}'.format(time_string(), api))


def visualize_sss_info(api, dataset, vis_save_dir):
  vis_save_dir = vis_save_dir.resolve()
  print ('{:} start to visualize {:} information'.format(time_string(), dataset))
  vis_save_dir.mkdir(parents=True, exist_ok=True)
  cache_file_path = vis_save_dir / '{:}-cache-sss-info.pth'.format(dataset)
  if not cache_file_path.exists():
    print ('Do not find cache file : {:}'.format(cache_file_path))
    params, flops, train_accs, valid_accs, test_accs = [], [], [], [], []
    for index in range(len(api)):
      info = api.get_cost_info(index, dataset)
      params.append(info['params'])
      flops.append(info['flops'])
      # accuracy
      info = api.get_more_info(index, dataset, hp='90')
      train_accs.append(info['train-accuracy'])
      test_accs.append(info['test-accuracy'])
      if dataset == 'cifar10':
        info = api.get_more_info(index, 'cifar10-valid', hp='90')
        valid_accs.append(info['valid-accuracy'])
      else:
        valid_accs.append(info['valid-accuracy'])
    info = {'params': params, 'flops': flops, 'train_accs': train_accs, 'valid_accs': valid_accs, 'test_accs': test_accs}
    torch.save(info, cache_file_path)
  else:
    print ('Find cache file : {:}'.format(cache_file_path))
    info = torch.load(cache_file_path)
    params, flops, train_accs, valid_accs, test_accs = info['params'], info['flops'], info['train_accs'], info['valid_accs'], info['test_accs']
  print ('{:} collect data done.'.format(time_string()))

  pyramid = ['8:16:32:48:64', '8:8:16:32:48', '8:8:16:16:32', '8:8:16:16:48', '8:8:16:16:64', '16:16:32:32:64', '32:32:64:64:64']
  pyramid_indexes = [api.query_index_by_arch(x) for x in pyramid]
  largest_indexes = [api.query_index_by_arch('64:64:64:64:64')]

  indexes = list(range(len(params)))
  dpi, width, height = 250, 8500, 1300
  figsize = width / float(dpi), height / float(dpi)
  LabelSize, LegendFontsize = 24, 24
  # resnet_scale, resnet_alpha = 120, 0.5
  xscale, xalpha = 120, 0.8

  fig, axs = plt.subplots(1, 4, figsize=figsize)
  # ax1, ax2, ax3, ax4, ax5 = axs
  for ax in axs:
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(LabelSize)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(LabelSize)
  ax2, ax3, ax4, ax5 = axs
  # ax1.xaxis.set_ticks(np.arange(0, max(indexes), max(indexes)//5))
  # ax1.scatter(indexes, test_accs, marker='o', s=0.5, c='tab:blue')
  # ax1.set_xlabel('architecture ID', fontsize=LabelSize)
  # ax1.set_ylabel('test accuracy (%)', fontsize=LabelSize)

  ax2.scatter(params, train_accs, marker='o', s=0.5, c='tab:blue')
  ax2.scatter([params[x] for x in pyramid_indexes], [train_accs[x] for x in pyramid_indexes], marker='*', s=xscale, c='tab:orange', label='Pyramid Structure', alpha=xalpha)
  ax2.scatter([params[x] for x in largest_indexes], [train_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax2.set_xlabel('#parameters (MB)', fontsize=LabelSize)
  ax2.set_ylabel('train accuracy (%)', fontsize=LabelSize)
  ax2.legend(loc=4, fontsize=LegendFontsize)

  ax3.scatter(params, test_accs, marker='o', s=0.5, c='tab:blue')
  ax3.scatter([params[x] for x in pyramid_indexes], [test_accs[x] for x in pyramid_indexes], marker='*', s=xscale, c='tab:orange', label='Pyramid Structure', alpha=xalpha)
  ax3.scatter([params[x] for x in largest_indexes], [test_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax3.set_xlabel('#parameters (MB)', fontsize=LabelSize)
  ax3.set_ylabel('test accuracy (%)', fontsize=LabelSize)
  ax3.legend(loc=4, fontsize=LegendFontsize)

  ax4.scatter(flops, train_accs, marker='o', s=0.5, c='tab:blue')
  ax4.scatter([flops[x] for x in pyramid_indexes], [train_accs[x] for x in pyramid_indexes], marker='*', s=xscale, c='tab:orange', label='Pyramid Structure', alpha=xalpha)
  ax4.scatter([flops[x] for x in largest_indexes], [train_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax4.set_xlabel('#FLOPs (M)', fontsize=LabelSize)
  ax4.set_ylabel('train accuracy (%)', fontsize=LabelSize)
  ax4.legend(loc=4, fontsize=LegendFontsize)

  ax5.scatter(flops, test_accs, marker='o', s=0.5, c='tab:blue')
  ax5.scatter([flops[x] for x in pyramid_indexes], [test_accs[x] for x in pyramid_indexes], marker='*', s=xscale, c='tab:orange', label='Pyramid Structure', alpha=xalpha)
  ax5.scatter([flops[x] for x in largest_indexes], [test_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax5.set_xlabel('#FLOPs (M)', fontsize=LabelSize)
  ax5.set_ylabel('test accuracy (%)', fontsize=LabelSize)
  ax5.legend(loc=4, fontsize=LegendFontsize)

  save_path = vis_save_dir / 'sss-{:}.png'.format(dataset)
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))
  plt.close('all')


def visualize_tss_info(api, dataset, vis_save_dir):
  vis_save_dir = vis_save_dir.resolve()
  print ('{:} start to visualize {:} information'.format(time_string(), dataset))
  vis_save_dir.mkdir(parents=True, exist_ok=True)
  cache_file_path = vis_save_dir / '{:}-cache-tss-info.pth'.format(dataset)
  if not cache_file_path.exists():
    print ('Do not find cache file : {:}'.format(cache_file_path))
    params, flops, train_accs, valid_accs, test_accs = [], [], [], [], []
    for index in range(len(api)):
      info = api.get_cost_info(index, dataset)
      params.append(info['params'])
      flops.append(info['flops'])
      # accuracy
      info = api.get_more_info(index, dataset, hp='200')
      train_accs.append(info['train-accuracy'])
      test_accs.append(info['test-accuracy'])
      if dataset == 'cifar10':
        info = api.get_more_info(index, 'cifar10-valid', hp='200')
        valid_accs.append(info['valid-accuracy'])
      else:
        valid_accs.append(info['valid-accuracy'])
    info = {'params': params, 'flops': flops, 'train_accs': train_accs, 'valid_accs': valid_accs, 'test_accs': test_accs}
    torch.save(info, cache_file_path)
  else:
    print ('Find cache file : {:}'.format(cache_file_path))
    info = torch.load(cache_file_path)
    params, flops, train_accs, valid_accs, test_accs = info['params'], info['flops'], info['train_accs'], info['valid_accs'], info['test_accs']
  print ('{:} collect data done.'.format(time_string()))

  resnet = ['|nor_conv_3x3~0|+|none~0|nor_conv_3x3~1|+|skip_connect~0|none~1|skip_connect~2|']
  resnet_indexes = [api.query_index_by_arch(x) for x in resnet]
  largest_indexes = [api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|')]

  indexes = list(range(len(params)))
  dpi, width, height = 250, 8500, 1300
  figsize = width / float(dpi), height / float(dpi)
  LabelSize, LegendFontsize = 24, 24
  # resnet_scale, resnet_alpha = 120, 0.5
  xscale, xalpha = 120, 0.8

  fig, axs = plt.subplots(1, 4, figsize=figsize)
  # ax1, ax2, ax3, ax4, ax5 = axs
  for ax in axs:
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(LabelSize)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(LabelSize)
  ax2, ax3, ax4, ax5 = axs
  # ax1.xaxis.set_ticks(np.arange(0, max(indexes), max(indexes)//5))
  # ax1.scatter(indexes, test_accs, marker='o', s=0.5, c='tab:blue')
  # ax1.set_xlabel('architecture ID', fontsize=LabelSize)
  # ax1.set_ylabel('test accuracy (%)', fontsize=LabelSize)

  ax2.scatter(params, train_accs, marker='o', s=0.5, c='tab:blue')
  ax2.scatter([params[x] for x in resnet_indexes] , [train_accs[x] for x in  resnet_indexes], marker='*', s=xscale, c='tab:orange', label='ResNet', alpha=xalpha)
  ax2.scatter([params[x] for x in largest_indexes], [train_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax2.set_xlabel('#parameters (MB)', fontsize=LabelSize)
  ax2.set_ylabel('train accuracy (%)', fontsize=LabelSize)
  ax2.legend(loc=4, fontsize=LegendFontsize)

  ax3.scatter(params, test_accs, marker='o', s=0.5, c='tab:blue')
  ax3.scatter([params[x] for x in resnet_indexes] , [test_accs[x] for x in  resnet_indexes], marker='*', s=xscale, c='tab:orange', label='ResNet', alpha=xalpha)
  ax3.scatter([params[x] for x in largest_indexes], [test_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax3.set_xlabel('#parameters (MB)', fontsize=LabelSize)
  ax3.set_ylabel('test accuracy (%)', fontsize=LabelSize)
  ax3.legend(loc=4, fontsize=LegendFontsize)

  ax4.scatter(flops, train_accs, marker='o', s=0.5, c='tab:blue')
  ax4.scatter([flops[x] for x in  resnet_indexes], [train_accs[x] for x in  resnet_indexes], marker='*', s=xscale, c='tab:orange', label='ResNet', alpha=xalpha)
  ax4.scatter([flops[x] for x in largest_indexes], [train_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax4.set_xlabel('#FLOPs (M)', fontsize=LabelSize)
  ax4.set_ylabel('train accuracy (%)', fontsize=LabelSize)
  ax4.legend(loc=4, fontsize=LegendFontsize)

  ax5.scatter(flops, test_accs, marker='o', s=0.5, c='tab:blue')
  ax5.scatter([flops[x] for x in  resnet_indexes], [test_accs[x] for x in  resnet_indexes], marker='*', s=xscale, c='tab:orange', label='ResNet', alpha=xalpha)
  ax5.scatter([flops[x] for x in largest_indexes], [test_accs[x] for x in largest_indexes], marker='x', s=xscale, c='tab:green',  label='Largest Candidate', alpha=xalpha)
  ax5.set_xlabel('#FLOPs (M)', fontsize=LabelSize)
  ax5.set_ylabel('test accuracy (%)', fontsize=LabelSize)
  ax5.legend(loc=4, fontsize=LegendFontsize)

  save_path = vis_save_dir / 'tss-{:}.png'.format(dataset)
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))
  plt.close('all')


def test_issue_81_82(api):
  results = api.query_by_index(0, 'cifar10')
  results = api.query_by_index(0, 'cifar10-valid', hp='200')
  print(results.keys())
  print(results[888].get_eval('x-valid'))
  result_dict = api.get_more_info(index=0, dataset='cifar10-valid', iepoch=11, hp='200', is_random=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='NAS-Bench-X', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--save_dir',    type=str, default='output/NAS-BENCH-202', help='Folder to save checkpoints and log.')
  parser.add_argument('--check_N',     type=int, default=32768,  help='For safety.')
  # use for train the model
  args = parser.parse_args()

  api201 = NASBench201API(os.path.join(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_0-e61699.pth'), verbose=True)
  test_issue_81_82(api201)
  test_api(api201, False)
  api201 = NASBench201API(None, verbose=True)
  test_issue_81_82(api201)
  visualize_tss_info(api201, 'cifar10', Path('output/vis-nas-bench'))
  visualize_tss_info(api201, 'cifar100', Path('output/vis-nas-bench'))
  visualize_tss_info(api201, 'ImageNet16-120', Path('output/vis-nas-bench'))
  test_api(api201, False)

  api301 = NASBench301API(None, verbose=True)
  visualize_sss_info(api301, 'cifar10', Path('output/vis-nas-bench'))
  visualize_sss_info(api301, 'cifar100', Path('output/vis-nas-bench'))
  visualize_sss_info(api301, 'ImageNet16-120', Path('output/vis-nas-bench'))
  test_api(api301, True)

  # save_dir = '{:}/visual'.format(args.save_dir)
