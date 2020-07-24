###############################################################
# NAS-Bench-201, ICLR 2020 (https://arxiv.org/abs/2001.00326) #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/experimental/vis-bench-ws.py --search_space tss
# Usage: python exps/experimental/vis-bench-ws.py --search_space sss
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
from nas_201_api import NASBench201API, NASBench301API
from log_utils import time_string


def fetch_data(root_dir='./output/search', search_space='tss', dataset=None):
  ss_dir = '{:}-{:}'.format(root_dir, search_space)
  alg2name, alg2path = OrderedDict(), OrderedDict()
  seeds = [777]
  alg2name['GDAS'] = 'gdas-affine0_BN0-None'
  alg2name['RSPS'] = 'random-affine0_BN0-None'
  alg2name['DARTS (1st)'] = 'darts-v1-affine0_BN0-None'
  alg2name['ENAS'] = 'enas-affine0_BN0-None'
  """
  alg2name['DARTS (2nd)'] = 'darts-v2-affine1_BN0-None'
  alg2name['SETN'] = 'setn-affine1_BN0-None'
  """
  for alg, name in alg2name.items():
    alg2path[alg] = os.path.join(ss_dir, dataset, name, 'seed-{:}-last-info.pth')
  alg2data = OrderedDict()
  for alg, path in alg2path.items():
    alg2data[alg] = []
    for seed in seeds:
      xpath = path.format(seed)
      assert os.path.isfile(xpath), 'invalid path : {:}'.format(xpath)
      data = torch.load(xpath, map_location=torch.device('cpu'))
      data = torch.load(data['last_checkpoint'], map_location=torch.device('cpu'))
      alg2data[alg].append(data['genotypes'])
  return alg2data


y_min_s = {('cifar10', 'tss'): 90,
           ('cifar10', 'sss'): 92,
           ('cifar100', 'tss'): 65,
           ('cifar100', 'sss'): 65,
           ('ImageNet16-120', 'tss'): 36,
           ('ImageNet16-120', 'sss'): 40}

y_max_s = {('cifar10', 'tss'): 94.5,
           ('cifar10', 'sss'): 93.3,
           ('cifar100', 'tss'): 72,
           ('cifar100', 'sss'): 70,
           ('ImageNet16-120', 'tss'): 44,
           ('ImageNet16-120', 'sss'): 46}

def visualize_curve(api, vis_save_dir, search_space):
  vis_save_dir = vis_save_dir.resolve()
  vis_save_dir.mkdir(parents=True, exist_ok=True)

  dpi, width, height = 250, 5200, 1400
  figsize = width / float(dpi), height / float(dpi)
  LabelSize, LegendFontsize = 16, 16

  def sub_plot_fn(ax, dataset):
    alg2data = fetch_data(search_space=search_space, dataset=dataset)
    alg2accuracies = OrderedDict()
    epochs = 100
    colors = ['b', 'g', 'c', 'm', 'y']
    ax.set_xlim(0, epochs)
    # ax.set_ylim(y_min_s[(dataset, search_space)], y_max_s[(dataset, search_space)])
    for idx, (alg, data) in enumerate(alg2data.items()):
      print('plot alg : {:}'.format(alg))
      xs, accuracies = [], []
      for iepoch in range(epochs+1):
        structures, accs = [_[iepoch-1] for _ in data], []
        for structure in structures:
          info = api.get_more_info(structure, dataset=dataset, hp=90 if isinstance(api, NASBench301API) else 200, is_random=False)
          accs.append(info['test-accuracy'])
        accuracies.append(sum(accs)/len(accs))
        xs.append(iepoch)
      alg2accuracies[alg] = accuracies
      ax.plot(xs, accuracies, c=colors[idx], label='{:}'.format(alg))
      ax.set_xlabel('The searching epoch', fontsize=LabelSize)
      ax.set_ylabel('Test accuracy on {:}'.format(dataset), fontsize=LabelSize)
      ax.set_title('Searching results on {:}'.format(dataset), fontsize=LabelSize+4)
    ax.legend(loc=4, fontsize=LegendFontsize)

  fig, axs = plt.subplots(1, 3, figsize=figsize)
  datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
  for dataset, ax in zip(datasets, axs):
    sub_plot_fn(ax, dataset)
    print('sub-plot {:} on {:} done.'.format(dataset, search_space))
  save_path = (vis_save_dir / '{:}-ws-curve.png'.format(search_space)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))
  plt.close('all')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='NAS-Bench-X', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--save_dir',     type=str,   default='output/vis-nas-bench/nas-algos', help='Folder to save checkpoints and log.')
  parser.add_argument('--search_space', type=str,   default='tss', choices=['tss', 'sss'], help='Choose the search space.')
  args = parser.parse_args()

  save_dir = Path(args.save_dir)
  alg2data = fetch_data(search_space='tss', dataset='cifar10')

  if args.search_space == 'tss':
    api = NASBench201API(verbose=False)
  elif args.search_space == 'sss':
    api = NASBench301API(verbose=False)
  else:
    raise ValueError('Invalid search space : {:}'.format(args.search_space))
  visualize_curve(api, save_dir, args.search_space)
