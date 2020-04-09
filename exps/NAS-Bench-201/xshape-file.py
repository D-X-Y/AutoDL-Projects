###############################################################
# NAS-Bench-201, ICLR 2020 (https://arxiv.org/abs/2001.00326) #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01           #
###############################################################
# Usage: python exps/NAS-Bench-201/xshape-file.py --mode check
###############################################################
import os, sys, time, torch, argparse
from typing import List, Text, Dict, Any
from tqdm import tqdm
from collections import defaultdict
from copy    import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import dict2config, load_config
from procedures   import bench_evaluate_for_seed
from procedures   import get_machine_info
from datasets     import get_datasets
from log_utils    import Logger, AverageMeter, time_string, convert_secs2time


def obtain_valid_ckp(save_dir: Text, total: int):
  possible_seeds = [777, 888]
  seed2ckps = defaultdict(list)
  miss2ckps = defaultdict(list)
  for i in range(total):
    for seed in possible_seeds:
      path = os.path.join(save_dir, 'arch-{:06d}-seed-{:04d}.pth'.format(i, seed))
      if os.path.exists(path):
        seed2ckps[seed].append(i)
      else:
        miss2ckps[seed].append(i)
    """
    ckps = [x for x in save_dir.glob('arch-{:06d}-seed-*.pth'.format(i))]
    for ckp in ckps:
      seed = ckp.name.split('-seed-')[-1].split('.pth')[0]
      seed2ckps[int(seed)].append(i)
    """
  for seed, xlist in seed2ckps.items():
    print('[{:}] [seed={:}] has {:}/{:}'.format(save_dir, seed, len(xlist), total))
  return dict(seed2ckps), dict(miss2ckps)
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='NAS-Bench-X', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--mode',        type=str, required=True, choices=['check', 'copy'], help='The script mode.')
  parser.add_argument('--save_dir',    type=str, default='output/NAS-BENCH-202', help='Folder to save checkpoints and log.')
  parser.add_argument('--check_N',     type=int, default=32768,  help='For safety.')
  # use for train the model
  args = parser.parse_args()
  possible_configs = ['01', '12', '90']
  if args.mode == 'check':
    for config in possible_configs:
      cur_save_dir = '{:}/raw-data-{:}'.format(args.save_dir, config)
      seed2ckps, miss2ckps = obtain_valid_ckp(cur_save_dir, args.check_N)
      torch.save(dict(seed2ckps=seed2ckps, miss2ckps=miss2ckps), '{:}/meta-{:}.pth'.format(args.save_dir, config))
  
