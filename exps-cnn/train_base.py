import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from utils import AverageMeter, time_string, convert_secs2time
from utils import print_log, obtain_accuracy
from utils import Cutout, count_parameters_in_MB
from nas import DARTS_V1, DARTS_V2, NASNet, PNASNet, AmoebaNet, ENASNet
from nas import DMS_V1, DMS_F1, GDAS_CC
from meta_nas import META_V1, META_V2
from train_utils import main_procedure
from train_utils_imagenet import main_procedure_imagenet
from scheduler import load_config

models = {'DARTS_V1': DARTS_V1,
          'DARTS_V2': DARTS_V2,
          'NASNet'  : NASNet,
          'PNASNet' : PNASNet,
          'ENASNet' : ENASNet,
          'DMS_V1'  : DMS_V1,
          'DMS_F1'  : DMS_F1,
          'GDAS_CC' : GDAS_CC,
          'META_V1' : META_V1,
          'META_V2' : META_V2,
          'AmoebaNet' : AmoebaNet}


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data_path',         type=str,   help='Path to dataset')
parser.add_argument('--dataset',           type=str,   choices=['imagenet', 'cifar10', 'cifar100'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',              type=str,   choices=models.keys(), help='the searched model.')
# 
parser.add_argument('--grad_clip',      type=float, help='gradient clipping')
parser.add_argument('--model_config',   type=str  , help='the model configuration')
parser.add_argument('--init_channels',  type=int  , help='the initial number of channels')
parser.add_argument('--layers',         type=int  , help='the number of layers.')

# log
parser.add_argument('--workers',       type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--save_path',     type=str, help='Folder to save checkpoints and log.')
parser.add_argument('--print_freq',    type=int, help='print frequency (default: 200)')
parser.add_argument('--manualSeed',    type=int, help='manual seed')
args = parser.parse_args()

assert torch.cuda.is_available(), 'torch.cuda is not available'

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
cudnn.benchmark = True
cudnn.enabled   = True
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)


def main():

  # Init logger
  args.save_path = os.path.join(args.save_path, 'seed-{:}'.format(args.manualSeed))
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log-seed-{:}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("Torch  version : {}".format(torch.__version__), log)
  print_log("CUDA   version : {}".format(torch.version.cuda), log)
  print_log("cuDNN  version : {}".format(cudnn.version()), log)
  print_log("Num of GPUs    : {}".format(torch.cuda.device_count()), log)
  args.dataset = args.dataset.lower()

  config = load_config(args.model_config)
  genotype = models[args.arch]
  print_log('configuration : {:}'.format(config), log)
  print_log('genotype      : {:}'.format(genotype), log)
  # clear GPU cache
  torch.cuda.empty_cache()
  if args.dataset == 'imagenet':
    main_procedure_imagenet(config, args.data_path, args, genotype, args.init_channels, args.layers, log)
  else:
    main_procedure(config, args.dataset, args.data_path, args, genotype, args.init_channels, args.layers, log)
  log.close()


if __name__ == '__main__':
  main() 
