##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################
import random, argparse
from .share_args import add_shared_args

def obtain_basic_args():
  parser = argparse.ArgumentParser(description='Train a classification model on typical image classification datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--resume'      ,     type=str,                   help='Resume path.')
  parser.add_argument('--init_model'  ,     type=str,                   help='The initialization model path.')
  parser.add_argument('--model_config',     type=str,                   help='The path to the model configuration')
  parser.add_argument('--optim_config',     type=str,                   help='The path to the optimizer configuration')
  parser.add_argument('--procedure'   ,     type=str,                   help='The procedure basic prefix.')
  parser.add_argument('--model_source',     type=str,  default='normal',help='The source of model defination.')
  parser.add_argument('--extra_model_path', type=str,  default=None,    help='The extra model ckp file (help to indicate the searched architecture).')
  add_shared_args( parser )
  # Optimization options
  parser.add_argument('--batch_size',       type=int,  default=2,       help='Batch size for training.')
  args = parser.parse_args()

  if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)
  assert args.save_dir is not None, 'save-path argument can not be None'
  return args
