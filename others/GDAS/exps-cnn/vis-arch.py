import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from graphviz import Digraph

parser = argparse.ArgumentParser("Visualize the Networks")
parser.add_argument('--checkpoint', type=str,   help='The path to the checkpoint.')
parser.add_argument('--save_dir',   type=str,   help='The directory to save the network plot.')
args = parser.parse_args()


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
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


if __name__ == '__main__':
  checkpoint = args.checkpoint
  assert os.path.isfile(checkpoint), 'Invalid path for checkpoint : {:}'.format(checkpoint)
  checkpoint = torch.load( checkpoint, map_location='cpu' )
  genotypes  = checkpoint['genotypes']
  save_dir   = Path(args.save_dir)
  subs       = ['normal', 'reduce']
  for sub in subs:
    if not (save_dir / sub).exists():
      (save_dir / sub).mkdir(parents=True, exist_ok=True)

  for key, network in genotypes.items():
    save_path = str(save_dir / 'normal' / 'epoch-{:03d}'.format( int(key) ))
    print('save into {:}'.format(save_path))
    plot(network.normal, save_path)

    save_path = str(save_dir / 'reduce' / 'epoch-{:03d}'.format( int(key) ))
    print('save into {:}'.format(save_path))
    plot(network.reduce, save_path)
