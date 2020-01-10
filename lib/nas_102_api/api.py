##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
############################################################################################
# NAS-Bench-102: Extending the Scope of Reproducible Neural Architecture Search, ICLR 2020 #
############################################################################################
# NAS-Bench-102-v1_0-e61699.pth : 6219 architectures are trained once, 1621 architectures are trained twice, 7785 architectures are trained three times. `LESS` only supports CIFAR10-VALID.
#
#
#
import os, sys, copy, random, torch, numpy as np
from collections import OrderedDict, defaultdict


def print_information(information, extra_info=None, show=False):
  dataset_names = information.get_dataset_names()
  strings = [information.arch_str, 'datasets : {:}, extra-info : {:}'.format(dataset_names, extra_info)]
  def metric2str(loss, acc):
    return 'loss = {:.3f}, top1 = {:.2f}%'.format(loss, acc)

  for ida, dataset in enumerate(dataset_names):
    #flop, param, latency = information.get_comput_costs(dataset)
    metric = information.get_comput_costs(dataset)
    flop, param, latency = metric['flops'], metric['params'], metric['latency']
    str1 = '{:14s} FLOP={:6.2f} M, Params={:.3f} MB, latency={:} ms.'.format(dataset, flop, param, '{:.2f}'.format(latency*1000) if latency is not None and latency > 0 else None)
    train_info = information.get_metrics(dataset, 'train')
    if dataset == 'cifar10-valid':
      valid_info = information.get_metrics(dataset, 'x-valid')
      str2 = '{:14s} train : [{:}], valid : [{:}]'.format(dataset, metric2str(train_info['loss'], train_info['accuracy']), metric2str(valid_info['loss'], valid_info['accuracy']))
    elif dataset == 'cifar10':
      test__info = information.get_metrics(dataset, 'ori-test')
      str2 = '{:14s} train : [{:}], test  : [{:}]'.format(dataset, metric2str(train_info['loss'], train_info['accuracy']), metric2str(test__info['loss'], test__info['accuracy']))
    else:
      valid_info = information.get_metrics(dataset, 'x-valid')
      test__info = information.get_metrics(dataset, 'x-test')
      str2 = '{:14s} train : [{:}], valid : [{:}], test : [{:}]'.format(dataset, metric2str(train_info['loss'], train_info['accuracy']), metric2str(valid_info['loss'], valid_info['accuracy']), metric2str(test__info['loss'], test__info['accuracy']))
    strings += [str1, str2]
  if show: print('\n'.join(strings))
  return strings


class NASBench102API(object):

  def __init__(self, file_path_or_dict, verbose=True):
    if isinstance(file_path_or_dict, str):
      if verbose: print('try to create the NAS-Bench-102 api from {:}'.format(file_path_or_dict))
      assert os.path.isfile(file_path_or_dict), 'invalid path : {:}'.format(file_path_or_dict)
      file_path_or_dict = torch.load(file_path_or_dict)
    elif isinstance(file_path_or_dict, dict):
      file_path_or_dict = copy.deepcopy( file_path_or_dict )
    else: raise ValueError('invalid type : {:} not in [str, dict]'.format(type(file_path_or_dict)))
    assert isinstance(file_path_or_dict, dict), 'It should be a dict instead of {:}'.format(type(file_path_or_dict))
    keys = ('meta_archs', 'arch2infos', 'evaluated_indexes')
    for key in keys: assert key in file_path_or_dict, 'Can not find key[{:}] in the dict'.format(key)
    self.meta_archs = copy.deepcopy( file_path_or_dict['meta_archs'] )
    self.arch2infos_less = OrderedDict()
    self.arch2infos_full = OrderedDict()
    for xkey in sorted(list(file_path_or_dict['arch2infos'].keys())):
      all_info = file_path_or_dict['arch2infos'][xkey]
      self.arch2infos_less[xkey] = ArchResults.create_from_state_dict( all_info['less'] )
      self.arch2infos_full[xkey] = ArchResults.create_from_state_dict( all_info['full'] )
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

  def random(self):
    return random.randint(0, len(self.meta_archs)-1)

  def query_index_by_arch(self, arch):
    if isinstance(arch, str):
      if arch in self.archstr2index: arch_index = self.archstr2index[ arch ]
      else                         : arch_index = -1
    elif hasattr(arch, 'tostr'):
      if arch.tostr() in self.archstr2index: arch_index = self.archstr2index[ arch.tostr() ]
      else                                 : arch_index = -1
    else: arch_index = -1
    return arch_index

  def reload(self, archive_root, index):
    assert os.path.isdir(archive_root), 'invalid directory : {:}'.format(archive_root)
    xfile_path = os.path.join(archive_root, '{:06d}-FULL.pth'.format(index))
    assert 0 <= index < len(self.meta_archs), 'invalid index of {:}'.format(index)
    assert os.path.isfile(xfile_path), 'invalid data path : {:}'.format(xfile_path)
    xdata = torch.load(xfile_path)
    assert isinstance(xdata, dict) and 'full' in xdata and 'less' in xdata, 'invalid format of data in {:}'.format(xfile_path)
    self.arch2infos_less[index] = ArchResults.create_from_state_dict( xdata['less'] )
    self.arch2infos_full[index] = ArchResults.create_from_state_dict( xdata['full'] )
  
  def query_by_arch(self, arch, use_12epochs_result=False):
    if isinstance(arch, int):
      arch_index = arch
    else:
      arch_index = self.query_index_by_arch(arch)
    if arch_index == -1: return None # the following two lines are used to support few training epochs
    if use_12epochs_result: arch2infos = self.arch2infos_less
    else                  : arch2infos = self.arch2infos_full
    if arch_index in arch2infos:
      strings = print_information(arch2infos[ arch_index ], 'arch-index={:}'.format(arch_index))
      return '\n'.join(strings)
    else:
      print ('Find this arch-index : {:}, but this arch is not evaluated.'.format(arch_index))
      return None

  # query information with the training of 12 epochs or 200 epochs
  # if dataname is None, return the ArchResults
  # else, return a dict with all trials on that dataset (the key is the seed)
  def query_by_index(self, arch_index, dataname=None, use_12epochs_result=False):
    if use_12epochs_result: basestr, arch2infos = '12epochs' , self.arch2infos_less
    else                  : basestr, arch2infos = '200epochs', self.arch2infos_full
    assert arch_index in arch2infos, 'arch_index [{:}] does not in arch2info with {:}'.format(arch_index, basestr)
    archInfo = copy.deepcopy( arch2infos[ arch_index ] )
    if dataname is None: return archInfo
    else:
      assert dataname in archInfo.get_dataset_names(), 'invalid dataset-name : {:}'.format(dataname)
      info = archInfo.query(dataname)
      return info

  def query_meta_info_by_index(self, arch_index, use_12epochs_result=False):
    if use_12epochs_result: basestr, arch2infos = '12epochs' , self.arch2infos_less
    else                  : basestr, arch2infos = '200epochs', self.arch2infos_full
    assert arch_index in arch2infos, 'arch_index [{:}] does not in arch2info with {:}'.format(arch_index, basestr)
    archInfo = copy.deepcopy( arch2infos[ arch_index ] )
    return archInfo

  def find_best(self, dataset, metric_on_set, FLOP_max=None, Param_max=None, use_12epochs_result=False):
    if use_12epochs_result: basestr, arch2infos = '12epochs' , self.arch2infos_less
    else                  : basestr, arch2infos = '200epochs', self.arch2infos_full
    best_index, highest_accuracy = -1, None
    for i, idx in enumerate(self.evaluated_indexes):
      info = arch2infos[idx].get_comput_costs(dataset)
      flop, param, latency = info['flops'], info['params'], info['latency']
      if FLOP_max  is not None and flop  > FLOP_max : continue
      if Param_max is not None and param > Param_max: continue
      xinfo = arch2infos[idx].get_metrics(dataset, metric_on_set)
      loss, accuracy = xinfo['loss'], xinfo['accuracy']
      if best_index == -1:
        best_index, highest_accuracy = idx, accuracy
      elif highest_accuracy < accuracy:
        best_index, highest_accuracy = idx, accuracy
    return best_index, highest_accuracy

  # return the topology structure of the `index`-th architecture
  def arch(self, index):
    assert 0 <= index < len(self.meta_archs), 'invalid index : {:} vs. {:}.'.format(index, len(self.meta_archs))
    return copy.deepcopy(self.meta_archs[index])

  # obtain the trained weights of the `index`-th architecture on `dataset` with the seed of `seed`
  def get_net_param(self, index, dataset, seed, use_12epochs_result=False):
    if use_12epochs_result: basestr, arch2infos = '12epochs' , self.arch2infos_less
    else                  : basestr, arch2infos = '200epochs', self.arch2infos_full
    archresult = arch2infos[index]
    return archresult.get_net_param(dataset, seed)

  # obtain the metric for the `index`-th architecture
  def get_more_info(self, index, dataset, iepoch=None, use_12epochs_result=False, is_random=True):
    if use_12epochs_result: basestr, arch2infos = '12epochs' , self.arch2infos_less
    else                  : basestr, arch2infos = '200epochs', self.arch2infos_full
    archresult = arch2infos[index]
    if dataset == 'cifar10-valid':
      train_info = archresult.get_metrics(dataset, 'train'   , iepoch=iepoch, is_random=is_random)
      valid_info = archresult.get_metrics(dataset, 'x-valid' , iepoch=iepoch, is_random=is_random)
      try:
        test__info = archresult.get_metrics(dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
      except:
        test__info = None
      total      = train_info['iepoch'] + 1
      xifo = {'train-loss'    : train_info['loss'],
              'train-accuracy': train_info['accuracy'],
              'train-all-time': train_info['all_time'],
              'valid-loss'    : valid_info['loss'],
              'valid-accuracy': valid_info['accuracy'],
              'valid-all-time': valid_info['all_time'],
              'valid-per-time': None if valid_info['all_time'] is None else valid_info['all_time'] / total}
      if test__info is not None:
        xifo['test-loss']     = test__info['loss']
        xifo['test-accuracy'] = test__info['accuracy']
      return xifo
    else:
      train_info = archresult.get_metrics(dataset, 'train'   , iepoch=iepoch, is_random=is_random)
      if dataset == 'cifar10':
        test__info = archresult.get_metrics(dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
      else:
        test__info = archresult.get_metrics(dataset, 'x-test', iepoch=iepoch, is_random=is_random)
      try:
        valid_info = archresult.get_metrics(dataset, 'x-valid', iepoch=iepoch, is_random=is_random)
      except:
        valid_info = None
      xifo = {'train-loss'    : train_info['loss'],
              'train-accuracy': train_info['accuracy'],
              'test-loss'     : test__info['loss'],
              'test-accuracy' : test__info['accuracy']}
      if valid_info is not None:
        xifo['valid-loss'] = valid_info['loss']
        xifo['valid-accuracy'] = valid_info['accuracy']
      return xifo

  def show(self, index=-1):
    if index < 0: # show all architectures
      print(self)
      for i, idx in enumerate(self.evaluated_indexes):
        print('\n' + '-' * 10 + ' The ({:5d}/{:5d}) {:06d}-th architecture! '.format(i, len(self.evaluated_indexes), idx) + '-'*10)
        print('arch : {:}'.format(self.meta_archs[idx]))
        strings = print_information(self.arch2infos_full[idx])
        print('>' * 40 + ' 200 epochs ' + '>' * 40)
        print('\n'.join(strings))
        strings = print_information(self.arch2infos_less[idx])
        print('>' * 40 + '  12 epochs ' + '>' * 40)
        print('\n'.join(strings))
        print('<' * 40 + '------------' + '<' * 40)
    else:
      if 0 <= index < len(self.meta_archs):
        if index not in self.evaluated_indexes: print('The {:}-th architecture has not been evaluated or not saved.'.format(index))
        else:
          strings = print_information(self.arch2infos_full[index])
          print('>' * 40 + ' 200 epochs ' + '>' * 40)
          print('\n'.join(strings))
          strings = print_information(self.arch2infos_less[index])
          print('>' * 40 + '  12 epochs ' + '>' * 40)
          print('\n'.join(strings))
          print('<' * 40 + '------------' + '<' * 40)
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

    flops      = [result.flop for result in results]
    params     = [result.params for result in results]
    lantencies = [result.get_latency() for result in results]
    lantencies = [x for x in lantencies if x > 0]
    mean_latency = np.mean(lantencies) if len(lantencies) > 0 else None
    time_infos = defaultdict(list)
    for result in results:
      time_info = result.get_times()
      for key, value in time_info.items(): time_infos[key].append( value )
     
    info = {'flops'  : np.mean(flops),
            'params' : np.mean(params),
            'latency': mean_latency}
    for key, value in time_infos.items():
      if len(value) > 0 and value[0] is not None:
        info[key] = np.mean(value)
      else: info[key] = None
    return info

  def get_metrics(self, dataset, setname, iepoch=None, is_random=False):
    x_seeds = self.dataset_seed[dataset]
    results = [self.all_results[ (dataset, seed) ] for seed in x_seeds]
    infos   = defaultdict(list)
    for result in results:
      if setname == 'train':
        info = result.get_train(iepoch)
      else:
        info = result.get_eval(setname, iepoch)
      for key, value in info.items(): infos[key].append( value )
    return_info = dict()
    if is_random:
      index = random.randint(0, len(results)-1)
      for key, value in infos.items(): return_info[key] = value[index]
    else:
      for key, value in infos.items():
        if len(value) > 0 and value[0] is not None:
          return_info[key] = np.mean(value)
        else: return_info[key] = None
    return return_info

  def show(self, is_print=False):
    return print_information(self, None, is_print)

  def get_dataset_names(self):
    return list(self.dataset_seed.keys())

  def get_net_param(self, dataset, seed=None):
    if seed is None:
      x_seeds = self.dataset_seed[dataset]
      return {seed: self.all_results[(dataset, seed)].get_net_param() for seed in x_seeds}
    else:
      return self.all_results[(dataset, seed)].get_net_param()

  def query(self, dataset, seed=None):
    if seed is None:
      x_seeds = self.dataset_seed[dataset]
      return {seed: self.all_results[ (dataset, seed) ] for seed in x_seeds}
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
    self.train_acc1es = copy.deepcopy(train_accs)
    self.train_acc5es = None
    self.train_losses = copy.deepcopy(train_losses)
    self.train_times  = None
    self.arch_config  = copy.deepcopy(arch_config)
    self.params     = params
    self.flop       = flop
    self.seed       = seed
    self.epochs     = epochs
    self.latency    = latency
    # evaluation results
    self.reset_eval()

  def update_train_info(self, train_acc1es, train_acc5es, train_losses, train_times):
    self.train_acc1es = train_acc1es
    self.train_acc5es = train_acc5es
    self.train_losses = train_losses
    self.train_times  = train_times

  def reset_eval(self):
    self.eval_names  = []
    self.eval_acc1es = {}
    self.eval_times  = {}
    self.eval_losses = {}

  def update_latency(self, latency):
    self.latency = copy.deepcopy( latency )

  def update_eval(self, accs, losses, times):  # new version
    data_names = set([x.split('@')[0] for x in accs.keys()])
    for data_name in data_names:
      assert data_name not in self.eval_names, '{:} has already been added into eval-names'.format(data_name)
      self.eval_names.append( data_name )
      for iepoch in range(self.epochs):
        xkey = '{:}@{:}'.format(data_name, iepoch)
        self.eval_acc1es[ xkey ] = accs[ xkey ]
        self.eval_losses[ xkey ] = losses[ xkey ]
        self.eval_times [ xkey ] = times[ xkey ]

  def update_OLD_eval(self, name, accs, losses): # old version
    assert name not in self.eval_names, '{:} has already added'.format(name)
    self.eval_names.append( name )
    for iepoch in range(self.epochs):
      if iepoch in accs:
        self.eval_acc1es['{:}@{:}'.format(name,iepoch)] = accs[iepoch]
        self.eval_losses['{:}@{:}'.format(name,iepoch)] = losses[iepoch]

  def __repr__(self):
    num_eval = len(self.eval_names)
    set_name = '[' + ', '.join(self.eval_names) + ']'
    return ('{name}({xname}, arch={arch}, FLOP={flop:.2f}M, Param={param:.3f}MB, seed={seed}, {num_eval} eval-sets: {set_name})'.format(name=self.__class__.__name__, xname=self.name, arch=self.arch_config['arch_str'], flop=self.flop, param=self.params, seed=self.seed, num_eval=num_eval, set_name=set_name))

  def get_latency(self):
    if self.latency is None: return -1
    else: return sum(self.latency) / len(self.latency)

  def get_times(self):
    if self.train_times is not None and isinstance(self.train_times, dict):
      train_times = list( self.train_times.values() )
      time_info = {'T-train@epoch': np.mean(train_times), 'T-train@total': np.sum(train_times)}
      for name in self.eval_names:
        xtimes = [self.eval_times['{:}@{:}'.format(name,i)] for i in range(self.epochs)]
        time_info['T-{:}@epoch'.format(name)] = np.mean(xtimes)
        time_info['T-{:}@total'.format(name)] = np.sum(xtimes)
    else:
      time_info = {'T-train@epoch':                 None, 'T-train@total':               None }
      for name in self.eval_names:
        time_info['T-{:}@epoch'.format(name)] = None
        time_info['T-{:}@total'.format(name)] = None
    return time_info

  def get_eval_set(self):
    return self.eval_names

  def get_train(self, iepoch=None):
    if iepoch is None: iepoch = self.epochs-1
    assert 0 <= iepoch < self.epochs, 'invalid iepoch={:} < {:}'.format(iepoch, self.epochs)
    if self.train_times is not None:
      xtime = self.train_times[iepoch]
      atime = sum([self.train_times[i] for i in range(iepoch+1)])
    else: xtime, atime = None, None
    return {'iepoch'  : iepoch,
            'loss'    : self.train_losses[iepoch],
            'accuracy': self.train_acc1es[iepoch],
            'cur_time': xtime,
            'all_time': atime}

  def get_eval(self, name, iepoch=None):
    if iepoch is None: iepoch = self.epochs-1
    assert 0 <= iepoch < self.epochs, 'invalid iepoch={:} < {:}'.format(iepoch, self.epochs)
    if isinstance(self.eval_times,dict) and len(self.eval_times) > 0:
      xtime = self.eval_times['{:}@{:}'.format(name,iepoch)]
      atime = sum([self.eval_times['{:}@{:}'.format(name,i)] for i in range(iepoch+1)])
    else: xtime, atime = None, None
    return {'iepoch'  : iepoch,
            'loss'    : self.eval_losses['{:}@{:}'.format(name,iepoch)],
            'accuracy': self.eval_acc1es['{:}@{:}'.format(name,iepoch)],
            'cur_time': xtime,
            'all_time': atime}

  def get_net_param(self):
    return self.net_state_dict

  def get_config(self, str2structure):
    #return copy.deepcopy(self.arch_config)
    return {'name': 'infer.tiny', 'C': self.arch_config['channel'], \
            'N'   : self.arch_config['num_cells'], \
            'genotype': str2structure(self.arch_config['arch_str']), 'num_classes': self.arch_config['class_num']}

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
