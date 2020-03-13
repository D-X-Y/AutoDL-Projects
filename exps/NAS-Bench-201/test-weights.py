#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
########################################################
# python exps/NAS-Bench-201/test-weights.py --api_path $HOME/.torch/NAS-Bench-201-v1_0-e61699.pth
########################################################
import os, sys, time, glob, random, argparse
import numpy as np
import torch
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from nas_201_api  import NASBench201API as API
from utils import weight_watcher


def main(meta_file, weight_dir, save_dir):
  import pdb;
  pdb.set_trace()


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Analysis of NAS-Bench-201")
  parser.add_argument('--save_dir',   type=str, default='./output/search-cell-nas-bench-201/visuals', help='The base-name of folder to save checkpoints and log.')
  parser.add_argument('--api_path',   type=str, default=None, help='The path to the NAS-Bench-201 benchmark file.')
  parser.add_argument('--weight_dir', type=str, default=None, help='The directory path to the weights of every NAS-Bench-201 architecture.')
  args = parser.parse_args()

  save_dir = Path(args.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)
  meta_file = Path(args.api_path)
  weight_dir = Path(args.weight_dir)
  assert meta_file.exists(), 'invalid path for api : {:}'.format(meta_file)

  main(meta_file, weight_dir, save_dir)

