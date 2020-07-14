##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###################################################################
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale #
# required to install hpbandster ##################################
# pip install hpbandster         ##################################
###################################################################
# python exps/algos-v2/bohb.py --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3
###################################################################
import os, sys, time, random, argparse
from copy import deepcopy
from pathlib import Path
import torch
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger
from log_utils    import AverageMeter, time_string, convert_secs2time
from nas_201_api  import NASBench201API as API
from models       import CellStructure, get_search_spaces
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale, ICML 2018
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker


def get_topology_config_space(search_space, max_nodes=4):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs


def get_size_config_space(search_space):
  cs = ConfigSpace.ConfigurationSpace()
	import pdb; pdb.set_trace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs


def config2topology_func(max_nodes=4):
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

  def __init__(self, *args, convert_func=None, dataname=None, nas_bench=None, time_budget=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.convert_func   = convert_func
    self._dataname      = dataname
    self._nas_bench     = nas_bench
    self.time_budget    = time_budget
    self.seen_archs     = []
    self.sim_cost_time  = 0
    self.real_cost_time = 0
    self.is_end         = False

  def get_the_best(self):
    assert len(self.seen_archs) > 0
    best_index, best_acc = -1, None
    for arch_index in self.seen_archs:
      info = self._nas_bench.get_more_info(arch_index, self._dataname, None, hp='200', is_random=True)
      vacc = info['valid-accuracy']
      if best_acc is None or best_acc < vacc:
        best_acc = vacc
        best_index = arch_index
    assert best_index != -1
    return best_index

  def compute(self, config, budget, **kwargs):
    start_time = time.time()
    structure  = self.convert_func( config )
    arch_index = self._nas_bench.query_index_by_arch( structure )
    info       = self._nas_bench.get_more_info(arch_index, self._dataname, None, hp='200', is_random=True)
    cur_time   = info['train-all-time'] + info['valid-per-time']
    cur_vacc   = info['valid-accuracy']
    self.real_cost_time += (time.time() - start_time)
    if self.sim_cost_time + cur_time <= self.time_budget and not self.is_end:
      self.sim_cost_time += cur_time
      self.seen_archs.append( arch_index )
      return ({'loss': 100 - float(cur_vacc),
               'info': {'seen-arch'     : len(self.seen_archs),
                        'sim-test-time' : self.sim_cost_time,
                        'current-arch'  : arch_index}
            })
    else:
      self.is_end = True
      return ({'loss': 100,
               'info': {'seen-arch'     : len(self.seen_archs),
                        'sim-test-time' : self.sim_cost_time,
                        'current-arch'  : None}
            })


def main(xargs, api):
  torch.set_num_threads(4)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  logger.log('{:} use api : {:}'.format(time_string(), api))
  search_space = get_search_spaces(xargs.search_space, 'nas-bench-301')
  if xargs.search_space == 'tss':
  	cs = get_topology_config_space(xargs.max_nodes, search_space)
  	config2structure = config2topology_func(xargs.max_nodes)
  else:
  	cs = get_size_config_space(xargs.max_nodes, search_space)
    import pdb; pdb.set_trace()
  
  hb_run_id = '0'

  NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
  ns_host, ns_port = NS.start()
  num_workers = 1

  workers = []
  for i in range(num_workers):
    w = MyWorker(nameserver=ns_host, nameserver_port=ns_port, convert_func=config2structure, dataname=dataname, nas_bench=nas_bench, time_budget=xargs.time_budget, run_id=hb_run_id, id=i)
    w.run(background=True)
    workers.append(w)

  start_time = time.time()
  bohb = BOHB(configspace=cs,
            run_id=hb_run_id,
            eta=3, min_budget=12, max_budget=200,
            nameserver=ns_host,
            nameserver_port=ns_port,
            num_samples=xargs.num_samples,
            random_fraction=xargs.random_fraction, bandwidth_factor=xargs.bandwidth_factor,
            ping_interval=10, min_bandwidth=xargs.min_bandwidth)
  
  results = bohb.run(xargs.n_iters, min_n_workers=num_workers)

  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()

  real_cost_time = time.time() - start_time

  id2config = results.get_id2config_mapping()
  incumbent = results.get_incumbent_id()
  logger.log('Best found configuration: {:} within {:.3f} s'.format(id2config[incumbent]['config'], real_cost_time))
  best_arch = config2structure( id2config[incumbent]['config'] )

  info = nas_bench.query_by_arch(best_arch, '200')
  if info is None: logger.log('Did not find this architecture : {:}.'.format(best_arch))
  else           : logger.log('{:}'.format(info))
  logger.log('-'*100)

  logger.log('workers : {:.1f}s with {:} archs'.format(workers[0].time_budget, len(workers[0].seen_archs)))
  logger.close()
  return logger.log_dir, nas_bench.query_index_by_arch( best_arch ), real_cost_time
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("BOHB: Robust and Efficient Hyperparameter Optimization at Scale")
  parser.add_argument('--dataset',            type=str,  choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # general arg
  parser.add_argument('--search_space',       type=str,  choices=['tss', 'sss'], help='Choose the search space.')
  parser.add_argument('--time_budget',        type=int,  default=20000, help='The total time cost budge for searching (in seconds).')
  parser.add_argument('--loops_if_rand',      type=int,  default=500, help='The total runs for evaluation.')
  # BOHB
  parser.add_argument('--strategy', default="sampling",  type=str, nargs='?', help='optimization strategy for the acquisition function')
  parser.add_argument('--min_bandwidth',    default=.3,  type=float, nargs='?', help='minimum bandwidth for KDE')
  parser.add_argument('--num_samples',      default=64,  type=int, nargs='?', help='number of samples for the acquisition function')
  parser.add_argument('--random_fraction',  default=.33, type=float, nargs='?', help='fraction of random configurations')
  parser.add_argument('--bandwidth_factor', default=3,   type=int, nargs='?', help='factor multiplied to the bandwidth')
  parser.add_argument('--n_iters',          default=300, type=int, nargs='?', help='number of iterations for optimization method')
  # log
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  
  if args.search_space == 'tss':
    api = NASBench201API(verbose=False)
  elif args.search_space == 'sss':
    api = NASBench301API(verbose=False)
  else:
    raise ValueError('Invalid search space : {:}'.format(args.search_space))

  args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space), args.dataset, 'BOHB')
  print('save-dir : {:}'.format(args.save_dir))

  if args.rand_seed < 0:
    save_dir, all_info = None, collections.OrderedDict()
    for i in range(args.loops_if_rand):
      print ('{:} : {:03d}/{:03d}'.format(time_string(), i, args.loops_if_rand))
      args.rand_seed = random.randint(1, 100000)
      save_dir, all_archs, all_total_times = main(args, api)
      all_info[i] = {'all_archs': all_archs,
                     'all_total_times': all_total_times}
    save_path = save_dir / 'results.pth'
    print('save into {:}'.format(save_path))
    torch.save(all_info, save_path)
  else:
    main(args, api)
