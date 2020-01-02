# python ./exps/vis/test.py
import os, sys, random
from pathlib import Path
import torch
import numpy as np
from collections import OrderedDict
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from graphviz import Digraph


def test_nas_api():
  from nas_102_api import ArchResults
  xdata   = torch.load('/home/dxy/FOR-RELEASE/NAS-Projects/output/NAS-BENCH-102-4/simplifies/architectures/000157-FULL.pth')
  for key in ['full', 'less']:
    print ('\n------------------------- {:} -------------------------'.format(key))
    archRes = ArchResults.create_from_state_dict(xdata[key])
    print(archRes)
    print(archRes.arch_idx_str())
    print(archRes.get_dataset_names())
    print(archRes.get_comput_costs('cifar10-valid'))
    # get the metrics
    print(archRes.get_metrics('cifar10-valid', 'x-valid', None, False))
    print(archRes.get_metrics('cifar10-valid', 'x-valid', None,  True))
    print(archRes.query('cifar10-valid', 777))


OPS    = ['skip-connect', 'conv-1x1', 'conv-3x3', 'pool-3x3']
COLORS = ['chartreuse'  , 'cyan'    , 'navyblue', 'chocolate1']

def plot(filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  steps = 5
  for i in range(0, steps):
    if i == 0:
      g.node(str(i), fillcolor='darkseagreen2')
    elif i+1 == steps:
      g.node(str(i), fillcolor='palegoldenrod')
    else: g.node(str(i), fillcolor='lightblue')

  for i in range(1, steps):
    for xin in range(i):
      op_i = random.randint(0, len(OPS)-1)
      #g.edge(str(xin), str(i), label=OPS[op_i], fillcolor=COLORS[op_i])
      g.edge(str(xin), str(i), label=OPS[op_i], color=COLORS[op_i], fillcolor=COLORS[op_i])
      #import pdb; pdb.set_trace()
  g.render(filename, cleanup=True, view=False)


if __name__ == '__main__':
  test_nas_api()
  for i in range(200): plot('{:04d}'.format(i))
