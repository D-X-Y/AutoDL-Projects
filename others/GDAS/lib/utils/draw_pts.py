import os, sys, time
import numpy as np
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def draw_points(points, labels, save_path):
  title = 'the visualized features'
  dpi = 100 
  width, height = 1000, 1000
  legend_fontsize = 10
  figsize = width / float(dpi), height / float(dpi)
  fig = plt.figure(figsize=figsize)

  classes = np.unique(labels).tolist()
  colors = cm.rainbow(np.linspace(0, 1, len(classes)))

  legends = []
  legendnames = []

  for cls, c in zip(classes, colors):
    
    indexes = labels == cls
    ptss = points[indexes, :]
    x = ptss[:,0]
    y = ptss[:,1]
    if cls % 2 == 0: marker = 'x'
    else:            marker = 'o'
    legend = plt.scatter(x, y, color=c, s=1, marker=marker)
    legendname = '{:02d}'.format(cls+1)
    legends.append( legend )
    legendnames.append( legendname )

  plt.legend(legends, legendnames, scatterpoints=1, ncol=5, fontsize=8)

  if save_path is not None:
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print ('---- save figure {} into {}'.format(title, save_path))
  plt.close(fig)
