##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# python ./exps/vis/show-results.py --api_path ${HOME}/.torch/NAS-Bench-102-v1_0-e61699.pth
##################################################
import os, sys, argparse
from pathlib import Path
import torch
import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from nas_102_api import NASBench102API as API


def plot_results_nas(api, dataset, xset, root, file_name, y_lims):
  print ('root-path={:}, dataset={:}, xset={:}'.format(root, dataset, xset))
  checkpoints = ['./output/search-cell-nas-bench-102/R-EA-cifar10/results.pth',
                 './output/search-cell-nas-bench-102/REINFORCE-cifar10/results.pth',
                 './output/search-cell-nas-bench-102/RAND-cifar10/results.pth',
                 './output/search-cell-nas-bench-102/BOHB-cifar10/results.pth'
                ]
  legends, indexes = ['REA', 'REINFORCE', 'RANDOM', 'BOHB'], None
  All_Accs = OrderedDict()
  for legend, checkpoint in zip(legends, checkpoints):
    all_indexes = torch.load(checkpoint, map_location='cpu')
    accuracies  = []
    for x in all_indexes:
      info = api.arch2infos_full[ x ]
      metrics = info.get_metrics(dataset, xset, None, False)
      accuracies.append( metrics['accuracy'] )
    if indexes is None: indexes = list(range(len(all_indexes)))
    All_Accs[legend] = sorted(accuracies)
  
  color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
  dpi, width, height = 300, 3400, 2600
  LabelSize, LegendFontsize = 28, 28
  figsize = width / float(dpi), height / float(dpi)
  fig = plt.figure(figsize=figsize)
  x_axis = np.arange(0, 600)
  plt.xlim(0, max(indexes))
  plt.ylim(y_lims[0], y_lims[1])
  interval_x, interval_y = 100, y_lims[2]
  plt.xticks(np.arange(0, max(indexes), interval_x), fontsize=LegendFontsize)
  plt.yticks(np.arange(y_lims[0],y_lims[1], interval_y), fontsize=LegendFontsize)
  plt.grid()
  plt.xlabel('The index of runs', fontsize=LabelSize)
  plt.ylabel('The accuracy (%)', fontsize=LabelSize)

  for idx, legend in enumerate(legends):
    plt.plot(indexes, All_Accs[legend], color=color_set[idx], linestyle='-', label='{:}'.format(legend), lw=2)
    print ('{:} : mean = {:}, std = {:} :: {:.2f}$\\pm${:.2f}'.format(legend, np.mean(All_Accs[legend]), np.std(All_Accs[legend]), np.mean(All_Accs[legend]), np.std(All_Accs[legend])))
  plt.legend(loc=4, fontsize=LegendFontsize)
  save_path = root / '{:}-{:}-{:}'.format(dataset, xset, file_name)
  print('save figure into {:}\n'.format(save_path))
  fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', format='pdf')


def just_show(api):
  xtimes = {'RSPS': [8082.5, 7794.2, 8144.7],
            'DARTS-V1': [11582.1, 11347.0, 11948.2],
            'DARTS-V2': [35694.7, 36132.7, 35518.0],
            'GDAS'    : [31334.1, 31478.6, 32016.7],
            'SETN'    : [33528.8, 33831.5, 35058.3],
            'ENAS'    : [14340.2, 13817.3, 14018.9]}
  for xkey, xlist in xtimes.items():
    xlist = np.array(xlist)
    print ('{:4s} : mean-time={:.1f} s'.format(xkey, xlist.mean()))

  xpaths = {'RSPS'    : 'output/search-cell-nas-bench-102/RANDOM-NAS-cifar10/checkpoint/',
            'DARTS-V1': 'output/search-cell-nas-bench-102/DARTS-V1-cifar10/checkpoint/',
            'DARTS-V2': 'output/search-cell-nas-bench-102/DARTS-V2-cifar10/checkpoint/',
            'GDAS'    : 'output/search-cell-nas-bench-102/GDAS-cifar10/checkpoint/',
            'SETN'    : 'output/search-cell-nas-bench-102/SETN-cifar10/checkpoint/',
            'ENAS'    : 'output/search-cell-nas-bench-102/ENAS-cifar10/checkpoint/',
           }
  xseeds = {'RSPS'    : [5349, 59613, 5983],
            'DARTS-V1': [11416, 72873, 81184],
            'DARTS-V2': [43330, 79405, 79423],
            'GDAS'    : [19677, 884, 95950],
            'SETN'    : [20518, 61817, 89144],
            'ENAS'    : [30801, 75610, 97745],
           }

  def get_accs(xdata, index=-1):
    if index == -1:
      epochs = xdata['epoch']
      genotype = xdata['genotypes'][epochs-1]
      index = api.query_index_by_arch(genotype)
    pairs = [('cifar10-valid', 'x-valid'), ('cifar10', 'ori-test'), ('cifar100', 'x-valid'), ('cifar100', 'x-test'), ('ImageNet16-120', 'x-valid'), ('ImageNet16-120', 'x-test')]
    xresults = []
    for dataset, xset in pairs:
      metrics = api.arch2infos_full[index].get_metrics(dataset, xset, None, False)
      xresults.append( metrics['accuracy'] )
    return xresults

  for xkey in xpaths.keys():
    all_paths = [ '{:}/seed-{:}-basic.pth'.format(xpaths[xkey], seed) for seed in xseeds[xkey] ]
    all_datas = [torch.load(xpath) for xpath in all_paths]
    accyss = [get_accs(xdatas) for xdatas in all_datas]
    accyss = np.array( accyss )
    print('\nxkey = {:}'.format(xkey))
    for i in range(accyss.shape[1]): print('---->>>> {:.2f}$\\pm${:.2f}'.format(accyss[:,i].mean(), accyss[:,i].std()))

  print('\n{:}'.format(get_accs(None, 11472))) # resnet
  pairs = [('cifar10-valid', 'x-valid'), ('cifar10', 'ori-test'), ('cifar100', 'x-valid'), ('cifar100', 'x-test'), ('ImageNet16-120', 'x-valid'), ('ImageNet16-120', 'x-test')]
  for dataset, metric_on_set in pairs:
    arch_index, highest_acc = api.find_best(dataset, metric_on_set)
    print ('[{:10s}-{:10s} ::: index={:5d}, accuracy={:.2f}'.format(dataset, metric_on_set, arch_index, highest_acc))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='NAS-Bench-102', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--save_dir',  type=str, default='./output/search-cell-nas-bench-102/visuals', help='The base-name of folder to save checkpoints and log.')
  parser.add_argument('--api_path',  type=str, default=None,                                         help='The path to the NAS-Bench-102 benchmark file.')
  args = parser.parse_args()

  api  = API(args.api_path)

  root = Path(args.save_dir).resolve()
  root.mkdir(parents=True, exist_ok=True)

  just_show(api)
  """
  plot_results_nas(api, 'cifar10-valid' , 'x-valid' , root, 'nas-com.pdf', (85,95, 1))
  plot_results_nas(api, 'cifar10'       , 'ori-test', root, 'nas-com.pdf', (85,95, 1))
  plot_results_nas(api, 'cifar100'      , 'x-valid' , root, 'nas-com.pdf', (55,75, 3))
  plot_results_nas(api, 'cifar100'      , 'x-test'  , root, 'nas-com.pdf', (55,75, 3))
  plot_results_nas(api, 'ImageNet16-120', 'x-valid' , root, 'nas-com.pdf', (35,50, 3))
  plot_results_nas(api, 'ImageNet16-120', 'x-test'  , root, 'nas-com.pdf', (35,50, 3))
  """
