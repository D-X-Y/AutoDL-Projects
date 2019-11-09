import os, sys, time, random, argparse

def add_shared_args( parser ):
  # Data Generation
  parser.add_argument('--dataset',          type=str,                   help='The dataset name.')
  parser.add_argument('--data_path',        type=str,                   help='The dataset name.')
  parser.add_argument('--cutout_length',    type=int,                   help='The cutout length, negative means not use.')
  # Printing
  parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
  parser.add_argument('--print_freq_eval',  type=int,   default=100,    help='print frequency (default: 200)')
  # Checkpoints
  parser.add_argument('--eval_frequency',   type=int,   default=1,      help='evaluation frequency (default: 200)')
  parser.add_argument('--save_dir',         type=str,                   help='Folder to save checkpoints and log.')
  # Acceleration
  parser.add_argument('--workers',          type=int,   default=8,      help='number of data loading workers (default: 8)')
  # Random Seed
  parser.add_argument('--rand_seed',        type=int,   default=-1,     help='manual seed')
