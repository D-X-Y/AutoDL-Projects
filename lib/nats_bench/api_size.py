#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06 #
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size # 
#####################################################################################
# The history of benchmark files (the name is NATS-tss-[version]-[md5].pickle.pbz2) #
# [2020.08.28] NATS-tss-v1_0-50262.pickle.pbz2                                      #
#####################################################################################
import os, copy, random, numpy as np
from pathlib import Path
from typing import List, Text, Union, Dict, Optional
from collections import OrderedDict, defaultdict
from .api_utils import pickle_load
from .api_utils import ArchResults
from .api_utils import NASBenchMetaAPI
from .api_utils import remap_dataset_set_names


PICKLE_EXT = 'pickle.pbz2'
ALL_BASE_NAMES = ['NATS-tss-v1_0-50262']


def print_information(information, extra_info=None, show=False):
  dataset_names = information.get_dataset_names()
  strings = [information.arch_str, 'datasets : {:}, extra-info : {:}'.format(dataset_names, extra_info)]
  def metric2str(loss, acc):
    return 'loss = {:.3f} & top1 = {:.2f}%'.format(loss, acc)

  for ida, dataset in enumerate(dataset_names):
    metric = information.get_compute_costs(dataset)
    flop, param, latency = metric['flops'], metric['params'], metric['latency']
    str1 = '{:14s} FLOP={:6.2f} M, Params={:.3f} MB, latency={:} ms.'.format(dataset, flop, param, '{:.2f}'.format(latency*1000) if latency is not None and latency > 0 else None)
    train_info = information.get_metrics(dataset, 'train')
    if dataset == 'cifar10-valid':
      valid_info = information.get_metrics(dataset, 'x-valid')
      test__info = information.get_metrics(dataset, 'ori-test')
      str2 = '{:14s} train : [{:}], valid : [{:}], test : [{:}]'.format(
                dataset, metric2str(train_info['loss'], train_info['accuracy']),
                metric2str(valid_info['loss'], valid_info['accuracy']),
                metric2str(test__info['loss'], test__info['accuracy']))
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


"""
This is the class for the API of size search space in NATS-Bench.
"""
class NATSsize(NASBenchMetaAPI):

  """ The initialization function that takes the dataset file path (or a dict loaded from that path) as input. """
  def __init__(self, file_path_or_dict: Optional[Union[Text, Dict]]=None, fast_mode: bool=False, verbose: bool=True):
    self.filename = None
    self._search_space_name = 'size'
    self._fast_mode = fast_mode
    self._archive_dir = None
    self.reset_time()
    if file_path_or_dict is None:
      if self._fast_mode:
        self._archive_dir = os.path.join(os.environ['TORCH_HOME'], '{:}-simple'.format(ALL_BASE_NAMES[-1]))
      else:
        file_path_or_dict = os.path.join(os.environ['TORCH_HOME'], '{:}.{:}'.format(ALL_BASE_NAMES[-1], PICKLE_EXT))
      print ('Try to use the default NATS-Bench (size) path from fast_mode={:} and path={:}.'.format(self._fast_mode, file_path_or_dict))
    if isinstance(file_path_or_dict, str) or isinstance(file_path_or_dict, Path):
      file_path_or_dict = str(file_path_or_dict)
      if verbose:
        print('Try to create the NATS-Bench (size) api from {:} with fast_mode={:}'.format(file_path_or_dict, fast_mode))
      if not os.path.isfile(file_path_or_dict) and not os.path.isdir(file_path_or_dict):
        raise ValueError('{:} is neither a file or a dir.'.format(file_path_or_dict))
      self.filename = Path(file_path_or_dict).name
      if fast_mode:
        if os.path.isfile(file_path_or_dict):
          raise ValueError('fast_mode={:} must feed the path for directory : {:}'.format(fast_mode, file_path_or_dict))
        else:
          self._archive_dir = file_path_or_dict
      else:
        if os.path.isdir(file_path_or_dict):
          raise ValueError('fast_mode={:} must feed the path for file : {:}'.format(fast_mode, file_path_or_dict))
        else:
          file_path_or_dict = pickle_load(file_path_or_dict)
    elif isinstance(file_path_or_dict, dict):
      file_path_or_dict = copy.deepcopy(file_path_or_dict)
    self.verbose = verbose
    if isinstance(file_path_or_dict, dict):
      keys = ('meta_archs', 'arch2infos', 'evaluated_indexes')
      for key in keys: assert key in file_path_or_dict, 'Can not find key[{:}] in the dict'.format(key)
      self.meta_archs = copy.deepcopy(file_path_or_dict['meta_archs'])
      # This is a dict mapping each architecture to a dict, where the key is #epochs and the value is ArchResults
      self.arch2infos_dict = OrderedDict()
      self._avaliable_hps = set()
      for xkey in sorted(list(file_path_or_dict['arch2infos'].keys())):
        all_infos = file_path_or_dict['arch2infos'][xkey]
        hp2archres = OrderedDict()
        for hp_key, results in all_infos.items():
          hp2archres[hp_key] = ArchResults.create_from_state_dict(results)
          self._avaliable_hps.add(hp_key)  # save the avaliable hyper-parameter
        self.arch2infos_dict[xkey] = hp2archres
      self.evaluated_indexes = set(file_path_or_dict['evaluated_indexes'])
    elif self.archive_dir is not None:
      benchmark_meta = pickle_load('{:}/meta.{:}'.format(self.archive_dir, PICKLE_EXT))
      self.meta_archs = copy.deepcopy(benchmark_meta['meta_archs'])
      self.arch2infos_dict = OrderedDict()
      self._avaliable_hps = set()
      self.evaluated_indexes = set()
    else:
      raise ValueError('file_path_or_dict [{:}] must be a dict or archive_dir must be set'.format(type(file_path_or_dict)))
    self.archstr2index = {}
    for idx, arch in enumerate(self.meta_archs):
      assert arch not in self.archstr2index, 'This [{:}]-th arch {:} already in the dict ({:}).'.format(idx, arch, self.archstr2index[arch])
      self.archstr2index[arch] = idx
    if self.verbose:
      print('Create NATS-Bench (size) done with {:}/{:} architectures avaliable.'.format(len(self.evaluated_indexes), len(self.meta_archs)))

  def reload(self, archive_root: Text = None, index: int = None):
    """Overwrite all information of the 'index'-th architecture in the search space, where the data will be loaded from 'archive_root'.
       If index is None, overwrite all ckps.
    """
    if self.verbose:
      print('Call clear_params with archive_root={:} and index={:}'.format(archive_root, index))
    if archive_root is None:
      archive_root = os.path.join(os.environ['TORCH_HOME'], '{:}-full'.format(ALL_BASE_NAMES[-1]))
    assert os.path.isdir(archive_root), 'invalid directory : {:}'.format(archive_root)
    if index is None:
      indexes = list(range(len(self)))
    else:
      indexes = [index]
    for idx in indexes:
      assert 0 <= idx < len(self.meta_archs), 'invalid index of {:}'.format(idx)
      xfile_path = os.path.join(archive_root, '{:06d}.{:}'.format(idx, PICKLE_EXT))
      if not os.path.isfile(xfile_path):
        xfile_path = os.path.join(archive_root, '{:d}.{:}'.format(idx, PICKLE_EXT))
      assert os.path.isfile(xfile_path), 'invalid data path : {:}'.format(xfile_path)
      xdata = pickle_load(xfile_path)
      assert isinstance(xdata, dict), 'invalid format of data in {:}'.format(xfile_path)
      self.evaluated_indexes.add(idx)
      hp2archres = OrderedDict()
      for hp_key, results in xdata.items():
        hp2archres[hp_key] = ArchResults.create_from_state_dict(results)
        self._avaliable_hps.add(hp_key)
      self.arch2infos_dict[idx] = hp2archres

  def query_info_str_by_arch(self, arch, hp: Text='12'):
    """ This function is used to query the information of a specific architecture
        'arch' can be an architecture index or an architecture string
        When hp=01, the hyper-parameters used to train a model are in 'configs/nas-benchmark/hyper-opts/01E.config'
        When hp=12, the hyper-parameters used to train a model are in 'configs/nas-benchmark/hyper-opts/12E.config'
        When hp=90, the hyper-parameters used to train a model are in 'configs/nas-benchmark/hyper-opts/90E.config'
        The difference between these three configurations are the number of training epochs.
    """
    if self.verbose:
      print('Call query_info_str_by_arch with arch={:} and hp={:}'.format(arch, hp))
    return self._query_info_str_by_arch(arch, hp, print_information)

  def get_more_info(self, index, dataset: Text, iepoch=None, hp='12', is_random=True):
    """This function will return the metric for the `index`-th architecture
       `dataset` indicates the dataset:
          'cifar10-valid'  : using the proposed train set of CIFAR-10 as the training set
          'cifar10'        : using the proposed train+valid set of CIFAR-10 as the training set
          'cifar100'       : using the proposed train set of CIFAR-100 as the training set
          'ImageNet16-120' : using the proposed train set of ImageNet-16-120 as the training set
        `iepoch` indicates the index of training epochs from 0 to 11/199.
          When iepoch=None, it will return the metric for the last training epoch
          When iepoch=11, it will return the metric for the 11-th training epoch (starting from 0)
        `hp` indicates different hyper-parameters for training
          When hp=01, it trains the network with 01 epochs and the LR decayed from 0.1 to 0 within 01 epochs
          When hp=12, it trains the network with 01 epochs and the LR decayed from 0.1 to 0 within 12 epochs
          When hp=90, it trains the network with 01 epochs and the LR decayed from 0.1 to 0 within 90 epochs
        `is_random`
          When is_random=True, the performance of a random architecture will be returned
          When is_random=False, the performanceo of all trials will be averaged.
    """
    if self.verbose:
      print('Call the get_more_info function with index={:}, dataset={:}, iepoch={:}, hp={:}, and is_random={:}.'.format(index, dataset, iepoch, hp, is_random))
    index = self.query_index_by_arch(index)  # To avoid the input is a string or an instance of a arch object
    self._prepare_info(index)
    if index not in self.arch2infos_dict:
      raise ValueError('Did not find {:} from arch2infos_dict.'.format(index))
    archresult = self.arch2infos_dict[index][str(hp)]
    # if randomly select one trial, select the seed at first
    if isinstance(is_random, bool) and is_random:
      seeds = archresult.get_dataset_seeds(dataset)
      is_random = random.choice(seeds)
    # collect the training information
    train_info = archresult.get_metrics(dataset, 'train', iepoch=iepoch, is_random=is_random)
    total = train_info['iepoch'] + 1
    xinfo = {'train-loss'    : train_info['loss'],
             'train-accuracy': train_info['accuracy'],
             'train-per-time': train_info['all_time'] / total,
             'train-all-time': train_info['all_time']}
    # collect the evaluation information
    if dataset == 'cifar10-valid':
      valid_info = archresult.get_metrics(dataset, 'x-valid', iepoch=iepoch, is_random=is_random)
      try:
        test_info = archresult.get_metrics(dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
      except:
        test_info = None
      valtest_info = None
    else:
      try: # collect results on the proposed test set
        if dataset == 'cifar10':
          test_info = archresult.get_metrics(dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
        else:
          test_info = archresult.get_metrics(dataset, 'x-test', iepoch=iepoch, is_random=is_random)
      except:
        test_info = None
      try: # collect results on the proposed validation set
        valid_info = archresult.get_metrics(dataset, 'x-valid', iepoch=iepoch, is_random=is_random)
      except:
        valid_info = None
      try:
        if dataset != 'cifar10':
          valtest_info = archresult.get_metrics(dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
        else:
          valtest_info = None
      except:
        valtest_info = None
    if valid_info is not None:
      xinfo['valid-loss'] = valid_info['loss']
      xinfo['valid-accuracy'] = valid_info['accuracy']
      xinfo['valid-per-time'] = valid_info['all_time'] / total
      xinfo['valid-all-time'] = valid_info['all_time']
    if test_info is not None:
      xinfo['test-loss'] = test_info['loss']
      xinfo['test-accuracy'] = test_info['accuracy']
      xinfo['test-per-time'] = test_info['all_time'] / total
      xinfo['test-all-time'] = test_info['all_time']
    if valtest_info is not None:
      xinfo['valtest-loss'] = valtest_info['loss']
      xinfo['valtest-accuracy'] = valtest_info['accuracy']
      xinfo['valtest-per-time'] = valtest_info['all_time'] / total
      xinfo['valtest-all-time'] = valtest_info['all_time']
    return xinfo

  def show(self, index: int = -1) -> None:
    """
    This function will print the information of a specific (or all) architecture(s).

    :param index: If the index < 0: it will loop for all architectures and print their information one by one.
                  else: it will print the information of the 'index'-th architecture.
    :return: nothing
    """
    self._show(index, print_information)
