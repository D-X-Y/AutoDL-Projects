##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##############################################################################
# Random Search for Hyper-Parameter Optimization, JMLR 2012 ##################
##############################################################################
# python ./exps/algos-v2/random_wo_share.py --dataset cifar10 --search_space tss
##############################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_search_spaces
from nas_201_api  import NASBench201API, NASBench301API
from .regularized_ea import random_topology_func, random_size_func


def main(xargs, api):
  torch.set_num_threads(4)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  search_space = get_search_spaces(xargs.search_space, 'nas-bench-301')
  if xargs.search_space == 'tss':
    random_arch = random_topology_func(search_space)
  else:
    random_arch = random_size_func(search_space)

  x_start_time = time.time()
  logger.log('{:} use nas_bench : {:}'.format(time_string(), nas_bench))
  best_arch, best_acc, total_time_cost, history = None, -1, [], []
  while total_time_cost[-1] < xargs.time_budget:
    arch = random_arch()
    accuracy, _, _, total_cost = api.simulate_train_eval(arch, xargs.dataset, '12')
    total_time_cost.append(total_cost)
    history.append(arch)
    if best_arch is None or best_acc < accuracy:
      best_acc, best_arch = accuracy, arch
    logger.log('[{:03d}] : {:} : accuracy = {:.2f}%'.format(len(history), arch, accuracy))
  logger.log('{:} best arch is {:}, accuracy = {:.2f}%, visit {:} archs with {:.1f} s (real-cost = {:.3f} s).'.format(time_string(), best_arch, best_acc, len(history), total_time_cost, time.time()-x_start_time))
  
  info = api.query_info_str_by_arch(best_arch, '200' if xargs.search_space == 'tss' else '90')
  logger.log('{:}'.format(info))
  logger.log('-'*100)
  logger.close()
  return logger.log_dir, total_time_cost, history


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Random NAS")
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--search_space',       type=str,   choices=['tss', 'sss'], help='Choose the search space.')

  parser.add_argument('--time_budget',        type=int,   default=20000, help='The total time cost budge for searching (in seconds).')
  parser.add_argument('--loops_if_rand',      type=int,   default=500,   help='The total runs for evaluation.')
  # log
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--rand_seed',          type=int,   default=-1,    help='manual seed')
  args = parser.parse_args()
  
  if args.search_space == 'tss':
    api = NASBench201API(verbose=False)
  elif args.search_space == 'sss':
    api = NASBench301API(verbose=False)
  else:
    raise ValueError('Invalid search space : {:}'.format(args.search_space))

  args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space), args.dataset, 'RANDOM')
  print('save-dir : {:}'.format(args.save_dir))

  if args.rand_seed < 0:
    save_dir, all_info = None, {}
    for i in range(args.loops_if_rand):
      print ('{:} : {:03d}/{:03d}'.format(time_string(), i, args.loops_if_rand))
      args.rand_seed = random.randint(1, 100000)
      save_dir, all_archs, all_total_times = main(args, api)
      all_info[i] = {'all_archs': all_archs,
                     'all_total_times': all_total_times}
    save_path = save_dir / 'results.pth'
    print('save into {:}'.format(save_path))
    torch.save(all_info, save_path)
  else:
    main(args, api)
