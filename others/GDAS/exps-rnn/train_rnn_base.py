##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, gc, sys, math, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import multiprocessing
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
print ('lib-dir : {:}'.format(lib_dir))
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from utils import AverageMeter, time_string, time_file_str, convert_secs2time
from utils import print_log, obtain_accuracy
from utils import count_parameters_in_MB
from nas_rnn import DARTS_V1, DARTS_V2, GDAS
from train_rnn_utils import main_procedure
from scheduler import load_config

Networks = {'DARTS_V1': DARTS_V1,
            'DARTS_V2': DARTS_V2,
            'GDAS'    : GDAS}

parser = argparse.ArgumentParser("RNN")
parser.add_argument('--arch',              type=str, choices=Networks.keys(), help='the network architecture')
parser.add_argument('--config_path',       type=str, help='the training configure for the discovered model')
# log
parser.add_argument('--save_path',         type=str, help='Folder to save checkpoints and log.')
parser.add_argument('--print_freq',        type=int, help='print frequency (default: 200)')
parser.add_argument('--manualSeed',        type=int, help='manual seed')
parser.add_argument('--threads',           type=int, default=4, help='the number of threads')
args = parser.parse_args()

assert torch.cuda.is_available(), 'torch.cuda is not available'

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
cudnn.benchmark = True
cudnn.enabled   = True
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
torch.set_num_threads(args.threads)

def main():

  # Init logger
  args.save_path = os.path.join(args.save_path, 'seed-{:}'.format(args.manualSeed))
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log-seed-{:}-{:}.txt'.format(args.manualSeed, time_file_str())), 'w')
  print_log('save path : {:}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("Torch  version : {}".format(torch.__version__), log)
  print_log("CUDA   version : {}".format(torch.version.cuda), log)
  print_log("cuDNN  version : {}".format(cudnn.version()), log)
  print_log("Num of GPUs    : {}".format(torch.cuda.device_count()), log)
  print_log("Num of CPUs    : {}".format(multiprocessing.cpu_count()), log)

  config = load_config( args.config_path )
  genotype = Networks[ args.arch ]

  main_procedure(config, genotype, args.save_path, args.print_freq, log)
  log.close()


if __name__ == '__main__':
  main() 
