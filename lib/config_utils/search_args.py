import os, sys, time, random, argparse
from .share_args import add_shared_args


def obtain_search_args():
  parser = argparse.ArgumentParser(description='Train a classification model on typical image classification datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--resume'        ,   type=str,                   help='Resume path.')
  parser.add_argument('--model_config'  ,   type=str,                   help='The path to the model configuration')
  parser.add_argument('--optim_config'  ,   type=str,                   help='The path to the optimizer configuration')
  parser.add_argument('--split_path'    ,   type=str,                   help='The split file path.')
  #parser.add_argument('--arch_para_pure',   type=int,                   help='The architecture-parameter pure or not.')
  parser.add_argument('--gumbel_tau_max',   type=float,                 help='The maximum tau for Gumbel.')
  parser.add_argument('--gumbel_tau_min',   type=float,                 help='The minimum tau for Gumbel.')
  parser.add_argument('--procedure'     ,   type=str,                   help='The procedure basic prefix.')
  parser.add_argument('--FLOP_ratio'    ,   type=float,                 help='The expected FLOP ratio.')
  parser.add_argument('--FLOP_weight'   ,   type=float,                 help='The loss weight for FLOP.')
  parser.add_argument('--FLOP_tolerant' ,   type=float,                 help='The tolerant range for FLOP.')
  # ablation studies
  parser.add_argument('--ablation_num_select', type=int,                help='The number of randomly selected channels.')
  add_shared_args( parser )
  # Optimization options
  parser.add_argument('--batch_size'    ,   type=int,   default=2,      help='Batch size for training.')
  args = parser.parse_args()

  if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)
  assert args.save_dir is not None, 'save-path argument can not be None'
  assert args.gumbel_tau_max is not None and args.gumbel_tau_min is not None
  assert args.FLOP_tolerant is not None and args.FLOP_tolerant > 0, 'invalid FLOP_tolerant : {:}'.format(FLOP_tolerant)
  #assert args.arch_para_pure is not None, 'arch_para_pure is not None: {:}'.format(args.arch_para_pure)
  #args.arch_para_pure = bool(args.arch_para_pure)
  return args
