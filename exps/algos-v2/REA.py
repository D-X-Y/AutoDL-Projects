##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################################
# Regularized Evolution for Image Classifier Architecture Search #
##################################################################
# python ./exps/algos-v2/REA.py --dataset cifar10 --search_space tss --time_budget 12000 --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/algos-v2/REA.py --dataset cifar100 --search_space tss --time_budget 12000 --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/algos-v2/REA.py --dataset ImageNet16-120 --search_space tss --time_budget 12000 --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/algos-v2/REA.py --dataset cifar10 --search_space sss --time_budget 12000 --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/algos-v2/REA.py --dataset cifar100 --search_space sss --time_budget 12000 --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/algos-v2/REA.py --dataset ImageNet16-120 --search_space sss --time_budget 12000 --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
##################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from nas_201_api  import NASBench201API, NASBench301API
from models       import CellStructure, get_search_spaces


class Model(object):

  def __init__(self):
    self.arch = None
    self.accuracy = None
    
  def __str__(self):
    """Prints a readable version of this bitstring."""
    return '{:}'.format(self.arch)
  

# This function is to mimic the training and evaluatinig procedure for a single architecture `arch`.
# The time_cost is calculated as the total training time for a few (e.g., 12 epochs) plus the evaluation time for one epoch.
# For use_012_epoch_training = True, the architecture is trained for 12 epochs, with LR being decaded from 0.1 to 0.
#       In this case, the LR schedular is converged.
# For use_012_epoch_training = False, the architecture is planed to be trained for 200 epochs, but we early stop its procedure.
#       
def train_and_eval(arch, nas_bench, extra_info, dataname='cifar10-valid', use_012_epoch_training=True):

  if use_012_epoch_training and nas_bench is not None:
    arch_index = nas_bench.query_index_by_arch( arch )
    assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
    valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
    #_, valid_acc = info.get_metrics('cifar10-valid', 'x-valid' , 25, True) # use the validation accuracy after 25 training epochs
  elif not use_012_epoch_training and nas_bench is not None:
    # Please contact me if you want to use the following logic, because it has some potential issues.
    # Please use `use_012_epoch_training=False` for cifar10 only.
    # It did return values for cifar100 and ImageNet16-120, but it has some potential issues. (Please email me for more details)
    arch_index, nepoch = nas_bench.query_index_by_arch( arch ), 25
    assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
    xoinfo = nas_bench.get_more_info(arch_index, 'cifar10-valid', iepoch=None, hp='12')
    xocost = nas_bench.get_cost_info(arch_index, 'cifar10-valid', hp='200')
    info = nas_bench.get_more_info(arch_index, dataname, nepoch, hp='200', is_random=True) # use the validation accuracy after 25 training epochs, which is used in our ICLR submission (not the camera ready).
    cost = nas_bench.get_cost_info(arch_index, dataname, hp='200')
    # The following codes are used to estimate the time cost.
    # When we build NAS-Bench-201, architectures are trained on different machines and we can not use that time record.
    # When we create checkpoints for converged_LR, we run all experiments on 1080Ti, and thus the time for each architecture can be fairly compared.
    nums = {'ImageNet16-120-train': 151700, 'ImageNet16-120-valid': 3000,
            'cifar10-valid-train' : 25000,  'cifar10-valid-valid' : 25000,
            'cifar100-train'      : 50000,  'cifar100-valid'      : 5000}
    estimated_train_cost = xoinfo['train-per-time'] / nums['cifar10-valid-train'] * nums['{:}-train'.format(dataname)] / xocost['latency'] * cost['latency'] * nepoch
    estimated_valid_cost = xoinfo['valid-per-time'] / nums['cifar10-valid-valid'] * nums['{:}-valid'.format(dataname)] / xocost['latency'] * cost['latency']
    try:
      valid_acc, time_cost = info['valid-accuracy'], estimated_train_cost + estimated_valid_cost
    except:
      valid_acc, time_cost = info['valtest-accuracy'], estimated_train_cost + estimated_valid_cost
  else:
    # train a model from scratch.
    raise ValueError('NOT IMPLEMENT YET')
  return valid_acc, time_cost


def random_topology_func(op_names, max_nodes=4):
  # Return a random architecture
  def random_architecture():
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name  = random.choice( op_names )
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return random_architecture


def random_size_func(info):
  # Return a random architecture
  def random_architecture():
    channels = []
    for i in range(info['numbers']):
      channels.append(
        str(random.choice(info['candidates'])))
    return ':'.join(channels)
  return random_architecture


def mutate_topology_func(op_names):
  """Computes the architecture for a child of the given parent architecture.
  The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
  """
  def mutate_topology_func(parent_arch):
    child_arch = deepcopy( parent_arch )
    node_id = random.randint(0, len(child_arch.nodes)-1)
    node_info = list( child_arch.nodes[node_id] )
    snode_id = random.randint(0, len(node_info)-1)
    xop = random.choice( op_names )
    while xop == node_info[snode_id][0]:
      xop = random.choice( op_names )
    node_info[snode_id] = (xop, node_info[snode_id][1])
    child_arch.nodes[node_id] = tuple( node_info )
    return child_arch
  return mutate_topology_func


def mutate_size_func(info):
  """Computes the architecture for a child of the given parent architecture.
  The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
  """
  def mutate_size_func(parent_arch):
    child_arch = deepcopy(parent_arch)
    child_arch = child_arch.split(':')
    index = random.randint(0, len(child_arch)-1)
    child_arch[index] = str(random.choice(info['candidates']))
    return ':'.join(child_arch)
  return mutate_size_func


def regularized_evolution(cycles, population_size, sample_size, time_budget, random_arch, mutate_arch, api, dataset):
  """Algorithm for regularized evolution (i.e. aging evolution).
  
  Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
  Classifier Architecture Search".
  
  Args:
    cycles: the number of cycles the algorithm should run for.
    population_size: the number of individuals to keep in the population.
    sample_size: the number of individuals that should participate in each tournament.
    time_budget: the upper bound of searching cost

  Returns:
    history: a list of `Model` instances, representing all the models computed
        during the evolution experiment.
  """
  population = collections.deque()
  api.reset_time()
  history, total_time_cost = [], []  # Not used by the algorithm, only used to report results.

  # Initialize the population with random models.
  while len(population) < population_size:
    model = Model()
    model.arch = random_arch()
    model.accuracy, time_cost, total_cost = api.simulate_train_eval(model.arch, dataset, '12')
    # Append the info
    population.append(model)
    history.append(model)
    total_time_cost.append(total_cost)

  # Carry out evolution in cycles. Each cycle produces a model and removes another.
  while total_time_cost[-1] < time_budget:
    # Sample randomly chosen models from the current population.
    start_time, sample = time.time(), []
    while len(sample) < sample_size:
      # Inefficient, but written this way for clarity. In the case of neural
      # nets, the efficiency of this line is irrelevant because training neural
      # nets is the rate-determining step.
      candidate = random.choice(list(population))
      sample.append(candidate)

    # The parent is the best model in the sample.
    parent = max(sample, key=lambda i: i.accuracy)

    # Create the child model and store it.
    child = Model()
    child.arch = mutate_arch(parent.arch)
    child.accuracy, time_cost, total_cost = api.simulate_train_eval(model.arch, dataset, '12')
    # Append the info
    population.append(child)
    history.append(child)
    total_time_cost.append(total_cost)

    # Remove the oldest model.
    population.popleft()
  return history, total_time_cost


def main(xargs, api):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads(xargs.workers)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  search_space = get_search_spaces(xargs.search_space, 'nas-bench-301')
  if xargs.search_space == 'tss':
    random_arch = random_topology_func(search_space)
    mutate_arch = mutate_topology_func(search_space)
  else:
    random_arch = random_size_func(search_space)
    mutate_arch = mutate_size_func(search_space)

  x_start_time = time.time()
  logger.log('{:} use api : {:}'.format(time_string(), api))
  logger.log('-'*30 + ' start searching with the time budget of {:} s'.format(xargs.time_budget))
  history, total_times = regularized_evolution(xargs.ea_cycles, xargs.ea_population, xargs.ea_sample_size, xargs.time_budget, random_arch, mutate_arch, api, xargs.dataset)
  logger.log('{:} regularized_evolution finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).'.format(time_string(), len(history), total_times[-1], time.time()-x_start_time))
  best_arch = max(history, key=lambda i: i.accuracy)
  best_arch = best_arch.arch
  logger.log('{:} best arch is {:}'.format(time_string(), best_arch))
  
  info = api.query_info_str_by_arch(best_arch, '200' if xargs.search_space == 'tss' else '90')
  logger.log('{:}'.format(info))
  logger.log('-'*100)
  logger.close()
  return logger.log_dir, [api.query_index_by_arch(x.arch) for x in history], total_times


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--search_space',       type=str,   choices=['tss', 'sss'], help='Choose the search space.')
  # channels and number-of-cells
  parser.add_argument('--ea_cycles',          type=int,   help='The number of cycles in EA.')
  parser.add_argument('--ea_population',      type=int,   help='The population size in EA.')
  parser.add_argument('--ea_sample_size',     type=int,   help='The sample size in EA.')
  parser.add_argument('--time_budget',        type=int,   help='The total time cost budge for searching (in seconds).')
  parser.add_argument('--loops_if_rand',      type=int,   default=500, help='The total runs for evaluation.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   default='./output/search', help='Folder to save checkpoints and log.')
  parser.add_argument('--rand_seed',          type=int,   default=-1,   help='manual seed')
  args = parser.parse_args()

  if args.search_space == 'tss':
    api = NASBench201API(verbose=False)
  elif args.search_space == 'sss':
    api = NASBench301API(verbose=False)
  else:
    raise ValueError('Invalid search space : {:}'.format(args.search_space))

  args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space), args.dataset, 'R-EA-SS{:}'.format(args.ea_sample_size))
  print('save-dir : {:}'.format(args.save_dir))

  if args.rand_seed < 0:
    save_dir, all_info = None, {}
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
