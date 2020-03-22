#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
###############################################################################################
# Before run these commands, the files must be properly put.
# python exps/NAS-Bench-201/test-weights.py --base_path $HOME/.torch/NAS-Bench-201-v1_0-e61699
# python exps/NAS-Bench-201/test-weights.py --base_path $HOME/.torch/NAS-Bench-201-v1_1-096897 --dataset cifar10-valid --use_12 1 --use_valid 1
# bash ./scripts-search/NAS-Bench-201/test-weights.sh cifar10-valid 1
###############################################################################################
import os, gc, sys, math, argparse, psutil
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from nas_201_api import NASBench201API as API
from log_utils import time_string
from models import get_cell_based_tiny_net
from utils import weight_watcher


def get_cor(A, B):
  return float(np.corrcoef(A, B)[0,1])


def tostr(accdict, norms):
  xstr = []
  for key, accs in accdict.items():
    cor = get_cor(accs, norms)
    xstr.append('{:}: {:.3f}'.format(key, cor))
  return ' '.join(xstr)


def evaluate(api, weight_dir, data: str, use_12epochs_result: bool):
  print('\nEvaluate dataset={:}'.format(data))
  norms, process = [], psutil.Process(os.getpid())
  final_val_accs = OrderedDict({'cifar10': [], 'cifar100': [], 'ImageNet16-120': []})
  final_test_accs = OrderedDict({'cifar10': [], 'cifar100': [], 'ImageNet16-120': []})
  for idx in range(len(api)):
    # info = api.get_more_info(idx, data, use_12epochs_result=use_12epochs_result, is_random=False)
    # import pdb; pdb.set_trace()
    for key in ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']:
      info = api.get_more_info(idx, key, use_12epochs_result=False, is_random=False)
      if key == 'cifar10-valid':
        final_val_accs['cifar10'].append(info['valid-accuracy'])
      elif key == 'cifar10':
        final_test_accs['cifar10'].append(info['test-accuracy'])
      else:
        final_test_accs[key].append(info['test-accuracy'])
        final_val_accs[key].append(info['valid-accuracy'])
    config = api.get_net_config(idx, data)
    net = get_cell_based_tiny_net(config)
    api.reload(weight_dir, idx)
    params = api.get_net_param(idx, data, None, use_12epochs_result=use_12epochs_result)
    cur_norms = []
    for seed, param in params.items():
      with torch.no_grad():
        net.load_state_dict(param)
        _, summary = weight_watcher.analyze(net, alphas=False)
        cur_norms.append(-summary['lognorm'])
    cur_norm = float(np.mean(cur_norms))
    if math.isnan(cur_norm):
      print ('  IGNORE {:} due to nan.'.format(idx))
      continue
    norms.append(cur_norm)
    api.clear_params(idx, None)
    if idx % 200 == 199 or idx + 1 == len(api):
      head = '{:05d}/{:05d}'.format(idx, len(api))
      stem_val = tostr(final_val_accs, norms)
      stem_test = tostr(final_test_accs, norms)
      print('{:} {:} {:} with {:} epochs ({:.2f} MB memory)'.format(time_string(), head, data, 12 if use_12epochs_result else 200, process.memory_info().rss / 1e6))
      print('  [Valid] -->>  {:}'.format(stem_val))
      print('  [Test.] -->>  {:}'.format(stem_test))
      gc.collect()


def main(meta_file: str, weight_dir, save_dir, xdata, use_12epochs_result):
  api = API(meta_file)
  datasets = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
  print(time_string() + ' ' + '='*50)
  for data in datasets:
    nums = api.statistics(data, True)
    total = sum([k*v for k, v in nums.items()])
    print('Using 012 epochs, trained on {:20s} : {:} trials in total ({:}).'.format(data, total, nums))
  print(time_string() + ' ' + '='*50)
  for data in datasets:
    nums = api.statistics(data, False)
    total = sum([k*v for k, v in nums.items()])
    print('Using 200 epochs, trained on {:20s} : {:} trials in total ({:}).'.format(data, total, nums))
  print(time_string() + ' ' + '='*50)

  #evaluate(api, weight_dir, 'cifar10-valid', False, True)
  evaluate(api, weight_dir, xdata, use_12epochs_result)
  
  print('{:} finish this test.'.format(time_string()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Analysis of NAS-Bench-201")
  parser.add_argument('--save_dir',   type=str, default='./output/search-cell-nas-bench-201/visuals', help='The base-name of folder to save checkpoints and log.')
  parser.add_argument('--base_path',  type=str, default=None, help='The path to the NAS-Bench-201 benchmark file and weight dir.')
  parser.add_argument('--dataset'  ,  type=str, default=None, help='.')
  parser.add_argument('--use_12'   ,  type=int, default=None, help='.')
  args = parser.parse_args()

  save_dir = Path(args.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)
  meta_file = Path(args.base_path + '.pth')
  weight_dir = Path(args.base_path + '-archive')
  assert meta_file.exists(), 'invalid path for api : {:}'.format(meta_file)
  assert weight_dir.exists() and weight_dir.is_dir(), 'invalid path for weight dir : {:}'.format(weight_dir)

  main(str(meta_file), weight_dir, save_dir, args.dataset, bool(args.use_12))

