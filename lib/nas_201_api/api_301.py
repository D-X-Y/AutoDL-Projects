#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06 #
############################################################################################
# NAS-Bench-301, coming soon.
############################################################################################
# The history of benchmark files:
# [2020.06.30] NAS-Bench-301-v1_0
# 
import os, copy, random, torch, numpy as np
from pathlib import Path
from typing import List, Text, Union, Dict, Optional
from collections import OrderedDict, defaultdict
from .api_utils import ArchResults
from .api_utils import NASBenchMetaAPI
from .api_utils import remap_dataset_set_names


ALL_BENCHMARK_FILES = ['NAS-Bench-301-v1_0-363be7.pth']
ALL_ARCHIVE_DIRS = ['NAS-Bench-301-v1_0-archive']


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
This is the class for the API of NAS-Bench-301.
"""
class NASBench301API(NASBenchMetaAPI):

  """ The initialization function that takes the dataset file path (or a dict loaded from that path) as input. """
  def __init__(self, file_path_or_dict: Optional[Union[Text, Dict]]=None, verbose: bool=True):
    self.filename = None
    self.reset_time()
    if file_path_or_dict is None:
      file_path_or_dict = os.path.join(os.environ['TORCH_HOME'], ALL_BENCHMARK_FILES[-1])
    if isinstance(file_path_or_dict, str) or isinstance(file_path_or_dict, Path):
      file_path_or_dict = str(file_path_or_dict)
      if verbose: print('try to create the NAS-Bench-201 api from {:}'.format(file_path_or_dict))
      assert os.path.isfile(file_path_or_dict), 'invalid path : {:}'.format(file_path_or_dict)
      self.filename = Path(file_path_or_dict).name
      file_path_or_dict = torch.load(file_path_or_dict, map_location='cpu')
    elif isinstance(file_path_or_dict, dict):
      file_path_or_dict = copy.deepcopy( file_path_or_dict )
    else: raise ValueError('invalid type : {:} not in [str, dict]'.format(type(file_path_or_dict)))
    assert isinstance(file_path_or_dict, dict), 'It should be a dict instead of {:}'.format(type(file_path_or_dict))
    self.verbose = verbose # [TODO] a flag indicating whether to print more logs
    keys = ('meta_archs', 'arch2infos', 'evaluated_indexes')
    for key in keys: assert key in file_path_or_dict, 'Can not find key[{:}] in the dict'.format(key)
    self.meta_archs = copy.deepcopy( file_path_or_dict['meta_archs'] )
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
    self.evaluated_indexes = sorted(list(file_path_or_dict['evaluated_indexes']))
    self.archstr2index = {}
    for idx, arch in enumerate(self.meta_archs):
      assert arch not in self.archstr2index, 'This [{:}]-th arch {:} already in the dict ({:}).'.format(idx, arch, self.archstr2index[arch])
      self.archstr2index[ arch ] = idx
    if self.verbose:
      print('Create NAS-Bench-301 done with {:}/{:} architectures avaliable.'.format(len(self.evaluated_indexes), len(self.meta_archs)))

  def reload(self, archive_root: Text = None, index: int = None):
    """Overwrite all information of the 'index'-th architecture in the search space, where the data will be loaded from 'archive_root'.
       If index is None, overwrite all ckps.
    """
    if self.verbose:
      print('Call clear_params with archive_root={:} and index={:}'.format(archive_root, index))
    if archive_root is None:
      archive_root = os.path.join(os.environ['TORCH_HOME'], ALL_ARCHIVE_DIRS[-1])
    assert os.path.isdir(archive_root), 'invalid directory : {:}'.format(archive_root)
    if index is None:
      indexes = list(range(len(self)))
    else:
      indexes = [index]
    for idx in indexes:
      assert 0 <= idx < len(self.meta_archs), 'invalid index of {:}'.format(idx)
      xfile_path = os.path.join(archive_root, '{:06d}-FULL.pth'.format(idx))
      if not os.path.isfile(xfile_path):
        xfile_path = os.path.join(archive_root, '{:d}-FULL.pth'.format(idx))
      assert os.path.isfile(xfile_path), 'invalid data path : {:}'.format(xfile_path)
      xdata = torch.load(xfile_path, map_location='cpu')
      assert isinstance(xdata, dict), 'invalid format of data in {:}'.format(xfile_path)

      hp2archres = OrderedDict()
      for hp_key, results in xdata.items():
        hp2archres[hp_key] = ArchResults.create_from_state_dict(results)
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
