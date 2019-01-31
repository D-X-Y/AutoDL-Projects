# python ./exps-nas/cvpr-vis.py --save_dir ./snapshots/NAS-VIS/
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from nas import DMS_V1, DMS_F1
from nas_rnn import DARTS_V2, GDAS
from graphviz import Digraph

parser = argparse.ArgumentParser("Visualize the Networks")
parser.add_argument('--save_dir',   type=str,   help='The directory to save the network plot.')
args = parser.parse_args()


def plot_cnn(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0, '{:}'.format(genotype)
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j, weight = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=False)

def plot_rnn(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("x_{t}", fillcolor='darkseagreen2')
  g.node("h_{t-1}", fillcolor='darkseagreen2')
  g.node("0", fillcolor='lightblue')
  g.edge("x_{t}", "0", fillcolor="gray")
  g.edge("h_{t-1}", "0", fillcolor="gray")
  steps = len(genotype)

  for i in range(1, steps + 1):
    g.node(str(i), fillcolor='lightblue')

  for i, (op, j) in enumerate(genotype):
    g.edge(str(j), str(i + 1), label=op, fillcolor="gray")

  g.node("h_{t}", fillcolor='palegoldenrod')
  for i in range(1, steps + 1):
    g.edge(str(i), "h_{t}", fillcolor="gray")

  g.render(filename, view=False)


if __name__ == '__main__':
  save_dir   = Path(args.save_dir)

  save_path = str(save_dir / 'DMS_V1-normal')
  plot_cnn(DMS_V1.normal, save_path)
  save_path = str(save_dir / 'DMS_V1-reduce')
  plot_cnn(DMS_V1.reduce, save_path)
  save_path = str(save_dir / 'DMS_F1-normal')
  plot_cnn(DMS_F1.normal, save_path)

  save_path = str(save_dir / 'DARTS-V2-RNN')
  plot_rnn(DARTS_V2.recurrent, save_path)

  save_path = str(save_dir / 'GDAS-V1-RNN')
  plot_rnn(GDAS.recurrent, save_path)
