##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# required to install hpbandster #################
##################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from nas_102_api  import NASBench102API as API
from models       import CellStructure, get_search_spaces
from R_EA import train_and_eval
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale, ICML 2018
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker


def get_configuration_space(max_nodes, search_space):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs


def config2structure_func(max_nodes):
  def config2structure(config):
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name = config[node_str]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return config2structure


class MyWorker(Worker):

  def __init__(self, *args, sleep_interval=0, convert_func=None, nas_bench=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.sleep_interval = sleep_interval
    self.convert_func   = convert_func
    self.nas_bench      = nas_bench
    self.test_time      = 0

  def compute(self, config, budget, **kwargs):
    structure = self.convert_func( config )
    reward, time_cost = train_and_eval(structure, self.nas_bench, None)
    import pdb; pdb.set_trace()
    self.test_time += 1
    return ({
            'loss': float(100-reward),
            'info': time_cost})


def main(xargs, nas_bench):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  assert xargs.dataset == 'cifar10', 'currently only support CIFAR-10'
  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
  cifar_split = load_config(split_Fpath, None, None)
  train_split, valid_split = cifar_split.train, cifar_split.valid
  logger.log('Load split file from {:}'.format(split_Fpath))
  config_path = 'configs/nas-benchmark/algos/R-EA.config'
  config = load_config(config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  # To split data
  train_data_v2 = deepcopy(train_data)
  train_data_v2.transform = valid_data.transform
  valid_data    = train_data_v2
  search_data   = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
  # data loader
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split) , num_workers=xargs.workers, pin_memory=True)
  valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=xargs.workers, pin_memory=True)
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
  extra_info = {'config': config, 'train_loader': train_loader, 'valid_loader': valid_loader}
  

  # nas dataset load
  assert xargs.arch_nas_dataset is not None and os.path.isfile(xargs.arch_nas_dataset)
  search_space = get_search_spaces('cell', xargs.search_space_name)
  cs = get_configuration_space(xargs.max_nodes, search_space)

  config2structure = config2structure_func(xargs.max_nodes)
  hb_run_id = '0'

  NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
  ns_host, ns_port = NS.start()
  num_workers = 1

  #nas_bench = AANASBenchAPI(xargs.arch_nas_dataset)
  #logger.log('{:} Create NAS-BENCH-API DONE'.format(time_string()))
  workers = []
  for i in range(num_workers):
    w = MyWorker(nameserver=ns_host, nameserver_port=ns_port, convert_func=config2structure, nas_bench=nas_bench, run_id=hb_run_id, id=i)
    w.run(background=True)
    workers.append(w)

  bohb = BOHB(configspace=cs,
            run_id=hb_run_id,
            eta=3, min_budget=3, max_budget=xargs.time_budget,
            nameserver=ns_host,
            nameserver_port=ns_port,
            num_samples=xargs.num_samples,
            random_fraction=xargs.random_fraction, bandwidth_factor=xargs.bandwidth_factor,
            ping_interval=10, min_bandwidth=xargs.min_bandwidth)
  #          optimization_strategy=xargs.strategy, num_samples=xargs.num_samples,
  
  results = bohb.run(xargs.n_iters, min_n_workers=num_workers)
  import pdb; pdb.set_trace()

  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()

  id2config = results.get_id2config_mapping()
  incumbent = results.get_incumbent_id()

  logger.log('Best found configuration: {:}'.format(id2config[incumbent]['config']))
  best_arch = config2structure( id2config[incumbent]['config'] )

  info = nas_bench.query_by_arch( best_arch )
  if info is None: logger.log('Did not find this architecture : {:}.'.format(best_arch))
  else           : logger.log('{:}'.format(info))
  logger.log('-'*100)

  logger.log('workers : {:}'.format(workers[0].test_time))
  logger.close()
  return logger.log_dir, nas_bench.query_index_by_arch( best_arch )
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--time_budget',        type=int,   help='The total time cost budge for searching (in seconds).')
  # BOHB
  parser.add_argument('--strategy', default="sampling", type=str, nargs='?', help='optimization strategy for the acquisition function')
  parser.add_argument('--min_bandwidth',    default=.3, type=float, nargs='?', help='minimum bandwidth for KDE')
  parser.add_argument('--num_samples',      default=64, type=int, nargs='?', help='number of samples for the acquisition function')
  parser.add_argument('--random_fraction',  default=.33, type=float, nargs='?', help='fraction of random configurations')
  parser.add_argument('--bandwidth_factor', default=3, type=int, nargs='?', help='factor multiplied to the bandwidth')
  parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  #if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.arch_nas_dataset is None or not os.path.isfile(args.arch_nas_dataset):
    nas_bench = None
  else:
    print ('{:} build NAS-Benchmark-API from {:}'.format(time_string(), args.arch_nas_dataset))
    nas_bench = API(args.arch_nas_dataset)
  if args.rand_seed < 0:
    save_dir, all_indexes, num = None, [], 500
    for i in range(num):
      print ('{:} : {:03d}/{:03d}'.format(time_string(), i, num))
      args.rand_seed = random.randint(1, 100000)
      save_dir, index = main(args, nas_bench)
      all_indexes.append( index ) 
    torch.save(all_indexes, save_dir / 'results.pth')
  else:
    main(args, nas_bench)
