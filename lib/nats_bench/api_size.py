#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 #
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
# The history of benchmark files are as follows,                             #
# where the format is (the name is NATS-sss-[version]-[md5].pickle.pbz2)     #
# [2020.08.31] NATS-sss-v1_0-50262.pickle.pbz2                               #
##############################################################################
# pylint: disable=line-too-long
"""The API for size search space in NATS-Bench."""
import collections
import copy
import os
import random
from typing import Dict, Optional, Text, Union, Any

from nats_bench.api_utils import ArchResults
from nats_bench.api_utils import NASBenchMetaAPI
from nats_bench.api_utils import nats_is_dir
from nats_bench.api_utils import nats_is_file
from nats_bench.api_utils import PICKLE_EXT
from nats_bench.api_utils import pickle_load
from nats_bench.api_utils import time_string


ALL_BASE_NAMES = ['NATS-sss-v1_0-50262']


def print_information(information, extra_info=None, show=False):
  """print out the information of a given ArchResults."""
  dataset_names = information.get_dataset_names()
  strings = [
      information.arch_str,
      'datasets : {:}, extra-info : {:}'.format(dataset_names, extra_info)
  ]

  def metric2str(loss, acc):
    return 'loss = {:.3f} & top1 = {:.2f}%'.format(loss, acc)

  for dataset in dataset_names:
    metric = information.get_compute_costs(dataset)
    flop, param, latency = metric['flops'], metric['params'], metric['latency']
    str1 = '{:14s} FLOP={:6.2f} M, Params={:.3f} MB, latency={:} ms.'.format(
        dataset, flop, param,
        '{:.2f}'.format(latency *
                        1000) if latency is not None and latency > 0 else None)
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
      str2 = '{:14s} train : [{:}], test  : [{:}]'.format(
          dataset, metric2str(train_info['loss'], train_info['accuracy']),
          metric2str(test__info['loss'], test__info['accuracy']))
    else:
      valid_info = information.get_metrics(dataset, 'x-valid')
      test__info = information.get_metrics(dataset, 'x-test')
      str2 = '{:14s} train : [{:}], valid : [{:}], test : [{:}]'.format(
          dataset, metric2str(train_info['loss'], train_info['accuracy']),
          metric2str(valid_info['loss'], valid_info['accuracy']),
          metric2str(test__info['loss'], test__info['accuracy']))
    strings += [str1, str2]
  if show: print('\n'.join(strings))
  return strings


class NATSsize(NASBenchMetaAPI):
  """This is the class for the API of size search space in NATS-Bench."""

  def __init__(self,
               file_path_or_dict: Optional[Union[Text, Dict[Text, Any]]] = None,
               fast_mode: bool = False,
               verbose: bool = True):
    """The initialization function that takes the dataset file path (or a dict loaded from that path) as input."""
    self._all_base_names = ALL_BASE_NAMES
    self.filename = None
    self._search_space_name = 'size'
    self._fast_mode = fast_mode
    self._archive_dir = None
    self._full_train_epochs = 90
    self.reset_time()
    if file_path_or_dict is None:
      if self._fast_mode:
        self._archive_dir = os.path.join(
            os.environ['TORCH_HOME'], '{:}-simple'.format(ALL_BASE_NAMES[-1]))
      else:
        file_path_or_dict = os.path.join(
            os.environ['TORCH_HOME'], '{:}.{:}'.format(
                ALL_BASE_NAMES[-1], PICKLE_EXT))
      print('{:} Try to use the default NATS-Bench (size) path from '
            'fast_mode={:} and path={:}.'.format(time_string(), self._fast_mode,
                                                 file_path_or_dict))
    if isinstance(file_path_or_dict, str):
      file_path_or_dict = str(file_path_or_dict)
      if verbose:
        print('{:} Try to create the NATS-Bench (size) api '
              'from {:} with fast_mode={:}'.format(
                  time_string(), file_path_or_dict, fast_mode))
      if not nats_is_file(file_path_or_dict) and not nats_is_dir(
          file_path_or_dict):
        raise ValueError('{:} is neither a file or a dir.'.format(
            file_path_or_dict))
      self.filename = os.path.basename(file_path_or_dict)
      if fast_mode:
        if nats_is_file(file_path_or_dict):
          raise ValueError('fast_mode={:} must feed the path for directory '
                           ': {:}'.format(fast_mode, file_path_or_dict))
        else:
          self._archive_dir = file_path_or_dict
      else:
        if nats_is_dir(file_path_or_dict):
          raise ValueError('fast_mode={:} must feed the path for file '
                           ': {:}'.format(fast_mode, file_path_or_dict))
        else:
          file_path_or_dict = pickle_load(file_path_or_dict)
    elif isinstance(file_path_or_dict, dict):
      file_path_or_dict = copy.deepcopy(file_path_or_dict)
    self.verbose = verbose
    if isinstance(file_path_or_dict, dict):
      keys = ('meta_archs', 'arch2infos', 'evaluated_indexes')
      for key in keys:
        if key not in file_path_or_dict:
          raise ValueError('Can not find key[{:}] in the dict'.format(key))
      self.meta_archs = copy.deepcopy(file_path_or_dict['meta_archs'])
      # NOTE(xuanyidong): This is a dict mapping each architecture to a dict,
      # where the key is #epochs and the value is ArchResults
      self.arch2infos_dict = collections.OrderedDict()
      self._avaliable_hps = set()
      for xkey in sorted(list(file_path_or_dict['arch2infos'].keys())):
        all_infos = file_path_or_dict['arch2infos'][xkey]
        hp2archres = collections.OrderedDict()
        for hp_key, results in all_infos.items():
          hp2archres[hp_key] = ArchResults.create_from_state_dict(results)
          self._avaliable_hps.add(hp_key)  # save the avaliable hyper-parameter
        self.arch2infos_dict[xkey] = hp2archres
      self.evaluated_indexes = set(file_path_or_dict['evaluated_indexes'])
    elif self.archive_dir is not None:
      benchmark_meta = pickle_load('{:}/meta.{:}'.format(
          self.archive_dir, PICKLE_EXT))
      self.meta_archs = copy.deepcopy(benchmark_meta['meta_archs'])
      self.arch2infos_dict = collections.OrderedDict()
      self._avaliable_hps = set()
      self.evaluated_indexes = set()
    else:
      raise ValueError('file_path_or_dict [{:}] must be a dict or archive_dir '
                       'must be set'.format(type(file_path_or_dict)))
    self.archstr2index = {}
    for idx, arch in enumerate(self.meta_archs):
      if arch in self.archstr2index:
        raise ValueError('This [{:}]-th arch {:} already in the '
                         'dict ({:}).'.format(
                             idx, arch, self.archstr2index[arch]))
      self.archstr2index[arch] = idx
    if self.verbose:
      print('{:} Create NATS-Bench (size) done with {:}/{:} architectures '
            'avaliable.'.format(time_string(),
                                len(self.evaluated_indexes),
                                len(self.meta_archs)))

  def query_info_str_by_arch(self, arch, hp: Text = '12'):
    """Query the information of a specific architecture.

    Args:
      arch: it can be an architecture index or an architecture string.

      hp: the hyperparamete indicator, could be 01, 12, or 90. The difference
          between these three configurations are the number of training epochs.

    Returns:
      ArchResults instance
    """
    if self.verbose:
      print('{:} Call query_info_str_by_arch with arch={:}'
            'and hp={:}'.format(time_string(), arch, hp))
    return self._query_info_str_by_arch(arch, hp, print_information)

  def get_more_info(self,
                    index,
                    dataset,
                    iepoch=None,
                    hp: Text = '12',
                    is_random: bool = True):
    """Return the metric for the `index`-th architecture.

    Args:
      index: the architecture index.
      dataset:
          'cifar10-valid'  : using the proposed train set of CIFAR-10 as the training set
          'cifar10'        : using the proposed train+valid set of CIFAR-10 as the training set
          'cifar100'       : using the proposed train set of CIFAR-100 as the training set
          'ImageNet16-120' : using the proposed train set of ImageNet-16-120 as the training set
      iepoch: the index of training epochs from 0 to 11/199.
          When iepoch=None, it will return the metric for the last training epoch
          When iepoch=11, it will return the metric for the 11-th training epoch (starting from 0)
      hp: indicates different hyper-parameters for training
          When hp=01, it trains the network with 01 epochs and the LR decayed from 0.1 to 0 within 01 epochs
          When hp=12, it trains the network with 01 epochs and the LR decayed from 0.1 to 0 within 12 epochs
          When hp=90, it trains the network with 01 epochs and the LR decayed from 0.1 to 0 within 90 epochs
      is_random:
          When is_random=True, the performance of a random architecture will be returned
          When is_random=False, the performanceo of all trials will be averaged.

    Returns:
      a dict, where key is the metric name and value is its value.
    """
    if self.verbose:
      print('{:} Call the get_more_info function with index={:}, dataset={:}, '
            'iepoch={:}, hp={:}, and is_random={:}.'.format(
                time_string(), index, dataset, iepoch, hp, is_random))
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
    train_info = archresult.get_metrics(
        dataset, 'train', iepoch=iepoch, is_random=is_random)
    total = train_info['iepoch'] + 1
    xinfo = {
        'train-loss': train_info['loss'],
        'train-accuracy': train_info['accuracy'],
        'train-per-time': train_info['all_time'] / total,
        'train-all-time': train_info['all_time']
    }
    # collect the evaluation information
    if dataset == 'cifar10-valid':
      valid_info = archresult.get_metrics(
          dataset, 'x-valid', iepoch=iepoch, is_random=is_random)
      try:
        test_info = archresult.get_metrics(
            dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
      except Exception as unused_e:  # pylint: disable=broad-except
        test_info = None
      valtest_info = None
    else:
      try:  # collect results on the proposed test set
        if dataset == 'cifar10':
          test_info = archresult.get_metrics(
              dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
        else:
          test_info = archresult.get_metrics(
              dataset, 'x-test', iepoch=iepoch, is_random=is_random)
      except Exception as unused_e:  # pylint: disable=broad-except
        test_info = None
      try:  # collect results on the proposed validation set
        valid_info = archresult.get_metrics(
            dataset, 'x-valid', iepoch=iepoch, is_random=is_random)
      except Exception as unused_e:  # pylint: disable=broad-except
        valid_info = None
      try:
        if dataset != 'cifar10':
          valtest_info = archresult.get_metrics(
              dataset, 'ori-test', iepoch=iepoch, is_random=is_random)
        else:
          valtest_info = None
      except Exception as unused_e:  # pylint: disable=broad-except
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
    """Print the information of a specific (or all) architecture(s)."""
    self._show(index, print_information)
