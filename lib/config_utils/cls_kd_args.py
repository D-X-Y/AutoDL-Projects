import os, sys, time, random, argparse
from .share_args import add_shared_args

def obtain_cls_kd_args():
  parser = argparse.ArgumentParser(description='Train a classification model on typical image classification datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--resume'      ,     type=str,                   help='Resume path.')
  parser.add_argument('--init_model'  ,     type=str,                   help='The initialization model path.')
  parser.add_argument('--model_config',     type=str,                   help='The path to the model configuration')
  parser.add_argument('--optim_config',     type=str,                   help='The path to the optimizer configuration')
  parser.add_argument('--procedure'   ,     type=str,                   help='The procedure basic prefix.')
  parser.add_argument('--KD_checkpoint',    type=str,                   help='The teacher checkpoint in knowledge distillation.')
  parser.add_argument('--KD_alpha'    ,     type=float,                 help='The alpha parameter in knowledge distillation.')
  parser.add_argument('--KD_temperature',   type=float,                 help='The temperature parameter in knowledge distillation.')
  #parser.add_argument('--KD_feature',       type=float,                 help='Knowledge distillation at the feature level.')
  add_shared_args( parser )
  # Optimization options
  parser.add_argument('--batch_size',       type=int,   default=2,      help='Batch size for training.')
  args = parser.parse_args()

  if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)
  assert args.save_dir is not None, 'save-path argument can not be None'
  return args
