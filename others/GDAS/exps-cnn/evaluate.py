##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# For evaluating the learned model
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
from nas import model_types as models
from train_utils import main_procedure
from train_utils_imagenet import main_procedure_imagenet
from scheduler import load_config


parser = argparse.ArgumentParser("Evaluate-CNN")
parser.add_argument('--data_path',         type=str,   help='Path to dataset.')
parser.add_argument('--checkpoint',        type=str,   help='Choose between Cifar10/100 and ImageNet.')
args = parser.parse_args()

assert torch.cuda.is_available(), 'torch.cuda is not available'


def main():

  assert os.path.isdir( args.data_path ), 'invalid data-path : {:}'.format(args.data_path)
  assert os.path.isfile( args.checkpoint ), 'invalid checkpoint : {:}'.format(args.checkpoint)

  checkpoint = torch.load( args.checkpoint )
  xargs      = checkpoint['args']
  config     = load_config(xargs.model_config)
  genotype   = models[xargs.arch]

  # clear GPU cache
  torch.cuda.empty_cache()
  if xargs.dataset == 'imagenet':
    main_procedure_imagenet(config, args.data_path, xargs, genotype, xargs.init_channels, xargs.layers, checkpoint['state_dict'], None)
  else:
    main_procedure(config, xargs.dataset, args.data_path, xargs, genotype, xargs.init_channels, xargs.layers, checkpoint['state_dict'], None)


if __name__ == '__main__':
  main() 
