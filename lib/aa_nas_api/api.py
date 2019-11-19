##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, copy, random, torch, numpy as np
from collections import OrderedDict


def print_information(information, extra_info=None, show=False):
  dataset_names = information.get_dataset_names()
  strings = [information.arch_str, 'datasets : {:}, extra-info : {:}'.format(dataset_names, extra_info)]
  def metric2str(loss, acc):
    return 'loss = {:.3f}, top1 = {:.2f}%'.format(loss, acc)

  for ida, dataset in enumerate(dataset_names):
    flop, param, latency = information.get_comput_costs(dataset)
    str1 = '{:14s} FLOP={:6.2f} M, Params={:.3f} MB, latency={:} ms.'.format(dataset, flop, param, '{:.2f}'.format(latency*1000) if latency > 0 else None)
    train_loss, train_acc = information.get_metrics(dataset, 'train')
    if dataset == 'cifar10-valid':
      valid_loss, valid_acc = information.get_metrics(dataset, 'x-valid')
      str2 = '{:14s} train : [{:}], valid : [{:}]'.format(dataset, metric2str(train_loss, train_acc), metric2str(valid_loss, valid_acc))
    elif dataset == 'cifar10':
      test__loss, test__acc = information.get_metrics(dataset, 'ori-test')
      str2 = '{:14s} train : [{:}], test  : [{:}]'.format(dataset, metric2str(train_loss, train_acc), metric2str(test__loss, test__acc))
    else:
      valid_loss, valid_acc = information.get_metrics(dataset, 'x-valid')
      test__loss, test__acc = information.get_metrics(dataset, 'x-test')
      str2 = '{:14s} train : [{:}], valid : [{:}], test : [{:}]'.format(dataset, metric2str(train_loss, train_acc), metric2str(valid_loss, valid_acc), metric2str(test__loss, test__acc))
    strings += [str1, str2]
  if show: print('\n'.join(strings))
  return strings


class AANASBenchAPI(object):

  def __init__(self, file_path_or_dict, verbose=True):
    if isinstance(file_path_or_dict, str):
      if verbose: print('try to create AA-NAS-Bench api from {:}'.format(file_path_or_dict))
      assert os.path.isfile(file_path_or_dict), 'invalid path : {:}'.format(file_path_or_dict)
      file_path_or_dict = torch.load(file_path_or_dict)
    else:
      file_path_or_dict = copy.deepcopy( file_path_or_dict )
    assert isinstance(file_path_or_dict, dict), 'It should be a dict instead of {:}'.format(type(file_path_or_dict))
    keys = ('meta_archs', 'arch2infos', 'evaluated_indexes')
    for key in keys: assert key in file_path_or_dict, 'Can not find key[{:}] in the dict'.format(key)
    self.meta_archs = copy.deepcopy( file_path_or_dict['meta_archs'] )
    self.arch2infos = OrderedDict()
    for xkey in sorted(list(file_path_or_dict['arch2infos'].keys())):
      self.arch2infos[xkey] = ArchResults.create_from_state_dict( file_path_or_dict['arch2infos'][xkey] )
    self.evaluated_indexes = sorted(list(file_path_or_dict['evaluated_indexes']))
    self.archstr2index = {}
    for idx, arch in enumerate(self.meta_archs):
      #assert arch.tostr() not in self.archstr2index, 'This [{:}]-th arch {:} already in the dict ({:}).'.format(idx, arch, self.archstr2index[arch.tostr()])
      assert arch not in self.archstr2index, 'This [{:}]-th arch {:} already in the dict ({:}).'.format(idx, arch, self.archstr2index[arch])
      self.archstr2index[ arch ] = idx

  def __getitem__(self, index):
    return copy.deepcopy( self.meta_archs[index] )

  def __len__(self):
    return len(self.meta_archs)

  def __repr__(self):
    return ('{name}({num}/{total} architectures)'.format(name=self.__class__.__name__, num=len(self.evaluated_indexes), total=len(self.meta_archs)))

  def query_index_by_arch(self, arch):
    if isinstance(arch, str):
      if arch in self.archstr2index: arch_index = self.archstr2index[ arch ]
      else                         : arch_index = -1
    elif hasattr(arch, 'tostr'):
      if arch.tostr() in self.archstr2index: arch_index = self.archstr2index[ arch.tostr() ]
      else                                 : arch_index = -1
    else: arch_index = -1
    return arch_index
  
  def query_by_arch(self, arch):
    arch_index = self.query_index_by_arch(arch)
    if arch_index == -1: return None
    if arch_index in self.arch2infos:
      strings = print_information(self.arch2infos[ arch_index ], 'arch-index={:}'.format(arch_index))
      return '\n'.join(strings)
    else:
      print ('Find this arch-index : {:}, but this arch is not evaluated.'.format(arch_index))
      return None

  def query_by_index(self, arch_index, dataname):
    assert arch_index in self.arch2infos, 'arch_index [{:}] does not in arch2info'.format(arch_index)
    archInfo = copy.deepcopy( self.arch2infos[ arch_index ] )
    assert dataname in archInfo.get_dataset_names(), 'invalid dataset-name : {:}'.format(dataname)
    info = archInfo.query(dataname)
    return info

  def query_meta_info_by_index(self, arch_index):
    assert arch_index in self.arch2infos, 'arch_index [{:}] does not in arch2info'.format(arch_index)
    archInfo = copy.deepcopy( self.arch2infos[ arch_index ] )
    return archInfo

  def find_best(self, dataset, metric_on_set, FLOP_max=None, Param_max=None):
    best_index, highest_accuracy = -1, None
    for i, idx in enumerate(self.evaluated_indexes):
      flop, param, latency = self.arch2infos[idx].get_comput_costs(dataset)
      if FLOP_max  is not None and flop  > FLOP_max : continue
      if Param_max is not None and param > Param_max: continue
      loss, accuracy = self.arch2infos[idx].get_metrics(dataset, metric_on_set)
      if best_index == -1:
        best_index, highest_accuracy = idx, accuracy
      elif highest_accuracy < accuracy:
        best_index, highest_accuracy = idx, accuracy
    return best_index

  def arch(self, index):
    assert 0 <= index < len(self.meta_archs), 'invalid index : {:} vs. {:}.'.format(index, len(self.meta_archs))
    return copy.deepcopy(self.meta_archs[index])

  def show(self, index=-1):
    if index == -1: # show all architectures
      print(self)
      for i, idx in enumerate(self.evaluated_indexes):
        print('\n' + '-' * 10 + ' The ({:5d}/{:5d}) {:06d}-th architecture! '.format(i, len(self.evaluated_indexes), idx) + '-'*10)
        print('arch : {:}'.format(self.meta_archs[idx]))
        strings = print_information(self.arch2infos[idx])
        print('>' * 20)
        print('\n'.join(strings))
        print('<' * 20)
    else:
      if 0 <= index < len(self.meta_archs):
        if index not in self.evaluated_indexes: print('The {:}-th architecture has not been evaluated or not saved.'.format(index))
        else:
          strings = print_information(self.arch2infos[index])
          print('\n'.join(strings))
      else:
        print('This index ({:}) is out of range (0~{:}).'.format(index, len(self.meta_archs)))



class ArchResults(object):

  def __init__(self, arch_index, arch_str):
    self.arch_index   = int(arch_index)
    self.arch_str     = copy.deepcopy(arch_str)
    self.all_results  = dict()
    self.dataset_seed = dict()
    self.clear_net_done = False

  def get_comput_costs(self, dataset):
    x_seeds = self.dataset_seed[dataset]
    results = [self.all_results[ (dataset, seed) ] for seed in x_seeds]
    flops   = [result.flop for result in results]
    params  = [result.params for result in results]
    lantencies = [result.get_latency() for result in results]
    return np.mean(flops), np.mean(params), np.mean(lantencies)

  def get_metrics(self, dataset, setname, iepoch=None, is_random=False):
    x_seeds = self.dataset_seed[dataset]
    results = [self.all_results[ (dataset, seed) ] for seed in x_seeds]
    loss, accuracy = [], []
    for result in results:
      if setname == 'train':
        info = result.get_train(iepoch)
      else:
        info = result.get_eval(setname, iepoch)
      loss.append( info['loss'] )
      accuracy.append( info['accuracy'] )
    if is_random:
      index = random.randint(0, len(loss)-1)
      return loss[index], accuracy[index]
    else:
      return float(np.mean(loss)), float(np.mean(accuracy))

  def show(self, is_print=False):
    return print_information(self, None, is_print)

  def get_dataset_names(self):
    return list(self.dataset_seed.keys())

  def query(self, dataset, seed=None):
    if seed is None:
      x_seeds = self.dataset_seed[dataset]
      return [self.all_results[ (dataset, seed) ] for seed in x_seeds]
    else:
      return self.all_results[ (dataset, seed) ]

  def arch_idx_str(self):
    return '{:06d}'.format(self.arch_index)

  def update(self, dataset_name, seed, result):
    if dataset_name not in self.dataset_seed:
      self.dataset_seed[dataset_name] = []
    assert seed not in self.dataset_seed[dataset_name], '{:}-th arch alreadly has this seed ({:}) on {:}'.format(self.arch_index, seed, dataset_name)
    self.dataset_seed[ dataset_name ].append( seed )
    self.dataset_seed[ dataset_name ] = sorted( self.dataset_seed[ dataset_name ] )
    assert (dataset_name, seed) not in self.all_results
    self.all_results[ (dataset_name, seed) ] = result
    self.clear_net_done = False

  def state_dict(self):
    state_dict = dict()
    for key, value in self.__dict__.items():
      if key == 'all_results': # contain the class of ResultsCount
        xvalue = dict()
        assert isinstance(value, dict), 'invalid type of value for {:} : {:}'.format(key, type(value))
        for _k, _v in value.items():
          assert isinstance(_v, ResultsCount), 'invalid type of value for {:}/{:} : {:}'.format(key, _k, type(_v))
          xvalue[_k] = _v.state_dict()
      else:
        xvalue = value
      state_dict[key] = xvalue
    return state_dict

  def load_state_dict(self, state_dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
      if key == 'all_results': # to convert to the class of ResultsCount
        xvalue = dict()
        assert isinstance(value, dict), 'invalid type of value for {:} : {:}'.format(key, type(value))
        for _k, _v in value.items():
          xvalue[_k] = ResultsCount.create_from_state_dict(_v)
      else: xvalue = value
      new_state_dict[key] = xvalue
    self.__dict__.update(new_state_dict)

  @staticmethod
  def create_from_state_dict(state_dict_or_file):
    x = ArchResults(-1, -1)
    if isinstance(state_dict_or_file, str): # a file path
      state_dict = torch.load(state_dict_or_file)
    elif isinstance(state_dict_or_file, dict):
      state_dict = state_dict_or_file
    else:
      raise ValueError('invalid type of state_dict_or_file : {:}'.format(type(state_dict_or_file)))
    x.load_state_dict(state_dict)
    return x

  def clear_params(self):
    for key, result in self.all_results.items():
      result.net_state_dict = None
    self.clear_net_done = True 

  def __repr__(self):
    return ('{name}(arch-index={index}, arch={arch}, {num} runs, clear={clear})'.format(name=self.__class__.__name__, index=self.arch_index, arch=self.arch_str, num=len(self.all_results), clear=self.clear_net_done))
    


class ResultsCount(object):

  def __init__(self, name, state_dict, train_accs, train_losses, params, flop, arch_config, seed, epochs, latency):
    self.name           = name
    self.net_state_dict = state_dict
    self.train_accs   = copy.deepcopy(train_accs)
    self.train_losses = copy.deepcopy(train_losses)
    self.arch_config  = copy.deepcopy(arch_config)
    self.params     = params
    self.flop       = flop
    self.seed       = seed
    self.epochs     = epochs
    self.latency    = latency
    # evaluation results
    self.reset_eval()

  def reset_eval(self):
    self.eval_names  = []
    self.eval_accs   = {}
    self.eval_losses = {}

  def update_latency(self, latency):
    self.latency = copy.deepcopy( latency )

  def get_latency(self):
    if self.latency is None: return -1
    else: return sum(self.latency) / len(self.latency)

  def update_eval(self, name, accs, losses):
    assert name not in self.eval_names, '{:} has already added'.format(name)
    self.eval_names.append( name )
    self.eval_accs[name] = copy.deepcopy( accs )
    self.eval_losses[name] = copy.deepcopy( losses )

  def __repr__(self):
    num_eval = len(self.eval_names)
    return ('{name}({xname}, arch={arch}, FLOP={flop:.2f}M, Param={param:.3f}MB, seed={seed}, {num_eval} eval-sets)'.format(name=self.__class__.__name__, xname=self.name, arch=self.arch_config['arch_str'], flop=self.flop, param=self.params, seed=self.seed, num_eval=num_eval))

  def valid_evaluation_set(self):
    return self.eval_names

  def get_train(self, iepoch=None):
    if iepoch is None: iepoch = self.epochs-1
    assert 0 <= iepoch < self.epochs, 'invalid iepoch={:} < {:}'.format(iepoch, self.epochs)
    return {'loss': self.train_losses[iepoch], 'accuracy': self.train_accs[iepoch]}

  def get_eval(self, name, iepoch=None):
    if iepoch is None: iepoch = self.epochs-1
    assert 0 <= iepoch < self.epochs, 'invalid iepoch={:} < {:}'.format(iepoch, self.epochs)
    return {'loss': self.eval_losses[name][iepoch], 'accuracy': self.eval_accs[name][iepoch]}

  def get_net_param(self):
    return self.net_state_dict

  def state_dict(self):
    _state_dict = {key: value for key, value in self.__dict__.items()}
    return _state_dict

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

  @staticmethod
  def create_from_state_dict(state_dict):
    x = ResultsCount(None, None, None, None, None, None, None, None, None, None)
    x.load_state_dict(state_dict)
    return x
