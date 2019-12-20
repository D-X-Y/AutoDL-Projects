# python ./exps/vis/test.py
import os, sys
from pathlib import Path
import torch
import numpy as np
from collections import OrderedDict
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))


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

if __name__ == '__main__':
  test_nas_api()
