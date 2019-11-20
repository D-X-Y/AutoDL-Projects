# python ./vis-exps/show-results.py
import os, sys
from pathlib import Path
import torch
import numpy as np
from collections import OrderedDict
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from aa_nas_api   import AANASBenchAPI

api = AANASBenchAPI('./output/AA-NAS-BENCH-4/simplifies/C16-N5-final-infos.pth')

def plot_results_nas(dataset, xset, file_name, y_lims):
  import matplotlib
  matplotlib.use('agg')
  import matplotlib.pyplot as plt
  root = Path('./output/cell-search-tiny-vis').resolve()
  print ('root path : {:}'.format( root ))
  root.mkdir(parents=True, exist_ok=True)
  checkpoints = ['./output/cell-search-tiny/R-EA-cifar10/results.pth',
                 './output/cell-search-tiny/REINFORCE-cifar10/results.pth',
                 './output/cell-search-tiny/RAND-cifar10/results.pth',
                 './output/cell-search-tiny/BOHB-cifar10/results.pth'
                ]
  legends, indexes = ['REA', 'REINFORCE', 'RANDOM', 'BOHB'], None
  All_Accs = OrderedDict()
  for legend, checkpoint in zip(legends, checkpoints):
    all_indexes = torch.load(checkpoint, map_location='cpu')
    accuracies  = []
    for x in all_indexes:
      info = api.arch2infos[ x ]
      _, accy = info.get_metrics(dataset, xset, None, False)
      accuracies.append( accy )
    if indexes is None: indexes = list(range(len(all_indexes)))
    All_Accs[legend] = sorted(accuracies)
  
  color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
  dpi, width, height = 300, 3400, 2600
  LabelSize, LegendFontsize = 26, 26
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
    print ('{:} : mean = {:}, std = {:}'.format(legend, np.mean(All_Accs[legend]), np.std(All_Accs[legend])))
  plt.legend(loc=4, fontsize=LegendFontsize)
  save_path = root / '{:}-{:}-{:}'.format(dataset, xset, file_name)
  print('save figure into {:}\n'.format(save_path))
  fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', format='pdf')


if __name__ == '__main__':
  plot_results_nas('cifar10', 'ori-test', 'nas-com.pdf', (85,95, 1))
  plot_results_nas('cifar100', 'x-valid', 'nas-com.pdf', (55,75, 3))
  plot_results_nas('cifar100', 'x-test' , 'nas-com.pdf', (55,75, 3))
  plot_results_nas('ImageNet16-120', 'x-valid', 'nas-com.pdf', (35,50, 3))
  plot_results_nas('ImageNet16-120', 'x-test' , 'nas-com.pdf', (35,50, 3))
