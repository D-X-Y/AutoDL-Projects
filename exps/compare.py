##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# python exps/compare.py --checkpoints basic.pth order.pth --names basic order --save ./output/vis/basic-vs-order.pdf
import sys, time, torch, random, argparse
from PIL     import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy    import deepcopy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

parser = argparse.ArgumentParser(description='Visualize the checkpoint and compare', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoints', type=str,    nargs='+',     help='checkpoint paths.')
parser.add_argument('--names',       type=str,    nargs='+',     help='names.')
parser.add_argument('--save',        type=str,                   help='the save path.')
args = parser.parse_args()


def visualize_acc(epochs, accuracies, names, save_path):

  LabelSize = 24
  LegendFontsize = 22
  matplotlib.rcParams['xtick.labelsize'] = LabelSize 
  matplotlib.rcParams['ytick.labelsize'] = LabelSize 
  color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
  dpi = 300
  width, height = 3400, 3600
  figsize = width / float(dpi), height / float(dpi)

  fig = plt.figure(figsize=figsize)
  plt.xlim(0, max(epochs))
  plt.ylim(0, 100)
  interval_x, interval_y = 20, 10
  plt.xticks(np.arange(0, max(epochs) + interval_x, interval_x), fontsize=LegendFontsize)
  plt.yticks(np.arange(0, 100 + interval_y, interval_y), fontsize=LegendFontsize)
  plt.grid()
  
  plt.xlabel('epoch', fontsize=16)
  plt.ylabel('accuracy (%)', fontsize=16)

  for idx, tag in enumerate(names):
    xaccs = [accuracies[idx][x] for x in epochs]
    plt.plot(epochs, xaccs, color=color_set[idx], linestyle='-', label='Test Accuracy : {:}'.format(tag), lw=3)
    plt.legend(loc=4, fontsize=LegendFontsize)
  
  if save_path is not None:
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    print ('---- save figure into {:}.'.format(save_path))
  plt.close(fig)


def main():
  checkpoints, names = args.checkpoints, args.names
  assert len(checkpoints) == len(names), 'invalid length : {:} vs {:}'.format(len(checkpoints), len(names))
  for i, checkpoint in enumerate(checkpoints):
    assert Path(checkpoint).exists(), 'The {:}-th checkpoint : {:} does not exist'.format( checkpoint )

  save_path = Path(args.save)
  save_dir  = save_path.parent
  save_dir.mkdir(parents=True, exist_ok=True)
  accuracies = []
  for checkpoint in checkpoints:
    checkpoint = torch.load( checkpoint )
    accuracies.append( checkpoint['valid_accuracies'] )
  epochs = [x for x in accuracies[0].keys() if isinstance(x, int)]
  epochs = sorted( epochs )
   
  visualize_acc(epochs, accuracies, names, save_path)


if __name__ == '__main__':
  main()
