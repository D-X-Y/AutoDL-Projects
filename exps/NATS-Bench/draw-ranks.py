###############################################################
# NATS-Bench (https://arxiv.org/pdf/2009.00437.pdf)           #
# The code to draw Figure 2 / 3 / 4 / 5 in our paper.         #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-ranks.py                 #
###############################################################
import os, sys, time, torch, argparse
import scipy
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
from log_utils import time_string
from models import get_cell_based_tiny_net
from nats_bench import create


def visualize_relative_info(api, vis_save_dir, indicator):
  vis_save_dir = vis_save_dir.resolve()
  # print ('{:} start to visualize {:} information'.format(time_string(), api))
  vis_save_dir.mkdir(parents=True, exist_ok=True)

  cifar010_cache_path = vis_save_dir / '{:}-cache-{:}-info.pth'.format('cifar10', indicator)
  cifar100_cache_path = vis_save_dir / '{:}-cache-{:}-info.pth'.format('cifar100', indicator)
  imagenet_cache_path = vis_save_dir / '{:}-cache-{:}-info.pth'.format('ImageNet16-120', indicator)
  cifar010_info = torch.load(cifar010_cache_path)
  cifar100_info = torch.load(cifar100_cache_path)
  imagenet_info = torch.load(imagenet_cache_path)
  indexes       = list(range(len(cifar010_info['params'])))

  print ('{:} start to visualize relative ranking'.format(time_string()))

  cifar010_ord_indexes = sorted(indexes, key=lambda i: cifar010_info['test_accs'][i])
  cifar100_ord_indexes = sorted(indexes, key=lambda i: cifar100_info['test_accs'][i])
  imagenet_ord_indexes = sorted(indexes, key=lambda i: imagenet_info['test_accs'][i])

  cifar100_labels, imagenet_labels = [], []
  for idx in cifar010_ord_indexes:
    cifar100_labels.append( cifar100_ord_indexes.index(idx) )
    imagenet_labels.append( imagenet_ord_indexes.index(idx) )
  print ('{:} prepare data done.'.format(time_string()))

  dpi, width, height = 200, 1400,  800
  figsize = width / float(dpi), height / float(dpi)
  LabelSize, LegendFontsize = 18, 12
  resnet_scale, resnet_alpha = 120, 0.5

  fig = plt.figure(figsize=figsize)
  ax  = fig.add_subplot(111)
  plt.xlim(min(indexes), max(indexes))
  plt.ylim(min(indexes), max(indexes))
  # plt.ylabel('y').set_rotation(30)
  plt.yticks(np.arange(min(indexes), max(indexes), max(indexes)//3), fontsize=LegendFontsize, rotation='vertical')
  plt.xticks(np.arange(min(indexes), max(indexes), max(indexes)//5), fontsize=LegendFontsize)
  ax.scatter(indexes, cifar100_labels, marker='^', s=0.5, c='tab:green', alpha=0.8)
  ax.scatter(indexes, imagenet_labels, marker='*', s=0.5, c='tab:red'  , alpha=0.8)
  ax.scatter(indexes, indexes        , marker='o', s=0.5, c='tab:blue' , alpha=0.8)
  ax.scatter([-1], [-1], marker='o', s=100, c='tab:blue' , label='CIFAR-10')
  ax.scatter([-1], [-1], marker='^', s=100, c='tab:green', label='CIFAR-100')
  ax.scatter([-1], [-1], marker='*', s=100, c='tab:red'  , label='ImageNet-16-120')
  plt.grid(zorder=0)
  ax.set_axisbelow(True)
  plt.legend(loc=0, fontsize=LegendFontsize)
  ax.set_xlabel('architecture ranking in CIFAR-10', fontsize=LabelSize)
  ax.set_ylabel('architecture ranking', fontsize=LabelSize)
  save_path = (vis_save_dir / '{:}-relative-rank.pdf'.format(indicator)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
  save_path = (vis_save_dir / '{:}-relative-rank.png'.format(indicator)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='NATS-Bench', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--save_dir',    type=str, default='output/vis-nas-bench/rank-stability', help='Folder to save checkpoints and log.')
  # use for train the model
  args = parser.parse_args()

  to_save_dir = Path(args.save_dir)

  # Figure 2
  visualize_relative_info(None, to_save_dir, 'tss')
  visualize_relative_info(None, to_save_dir, 'sss')