##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 ##########################
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
# pytest --capture=tee-sys                                                   #
##############################################################################
"""This file is used to quickly test the API."""
import os
import pytest
import random

from nats_bench.api_size import NATSsize
from nats_bench.api_size import ALL_BASE_NAMES as sss_base_names
from nats_bench.api_topology import NATStopology
from nats_bench.api_topology import ALL_BASE_NAMES as tss_base_names


def get_fake_torch_home_dir():
  print('This file is {:}'.format(os.path.abspath(__file__)))
  print('The current directory is {:}'.format(os.path.abspath(os.getcwd())))
  xname = 'FAKE_TORCH_HOME'
  if xname in os.environ:
    return os.environ['FAKE_TORCH_HOME']
  else:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fake_torch_dir')


class TestNATSBench(object):

  def test_nats_bench_tss(self, benchmark_dir=None, fake_random=True):
    if benchmark_dir is None:
      benchmark_dir = os.path.join(get_fake_torch_home_dir(), sss_base_names[-1] + '-simple')
    return _test_nats_bench(benchmark_dir, True, fake_random)

  def test_nats_bench_sss(self, benchmark_dir=None, fake_random=True):
    if benchmark_dir is None:
      benchmark_dir = os.path.join(get_fake_torch_home_dir(), tss_base_names[-1] + '-simple')
    return _test_nats_bench(benchmark_dir, False, fake_random)

  def prepare_fake_tss(self):
    print('')
    tss_benchmark_dir = os.path.join(get_fake_torch_home_dir(), tss_base_names[-1] + '-simple')
    api = NATStopology(tss_benchmark_dir, True, False)
    return api

  def test_01_th_issue(self):
    # Link: https://github.com/D-X-Y/NATS-Bench/issues/1
    api = self.prepare_fake_tss()
    # The performance of 0-th architecture on CIFAR-10 (trained by 12 epochs)
    info = api.get_more_info(0, 'cifar10', hp=12)
    # First of all, the data split in NATS-Bench is different from that in the official CIFAR paper.
    # In NATS-Bench, we split the original CIFAR-10 training set into two parts, i.e., a training set and a validation set.
    # In the following, we will use the splits of NATS-Bench to explain.
    print(info['comment'])
    print('The loss on the training + validation sets of CIFAR-10: {:}'.format(info['train-loss']))
    print('The total training time for 12 epochs on the training + validation sets of CIFAR-10: {:}'.format(info['train-all-time']))
    print('The per-epoch training time on CIFAR-10: {:}'.format(info['train-per-time']))
    print('The total evaluation time on the test set of CIFAR-10 for 12 times: {:}'.format(info['test-all-time']))
    print('The evaluation time on the test set of CIFAR-10: {:}'.format(info['test-per-time']))
    cost_info = api.get_cost_info(0, 'cifar10')
    xkeys = ['T-train@epoch',     # The per epoch training time on the training + validation sets of CIFAR-10.
             'T-train@total',
             'T-ori-test@epoch',  # The time cost for the evaluation on CIFAR-10 test set.
             'T-ori-test@total']  # T-ori-test@epoch * 12 times.
    for xkey in xkeys:
      print('The cost info [{:}] for 0-th architecture on CIFAR-10 is {:}'.format(xkey, cost_info[xkey]))
    
  def test_02_th_issue(self):
    # https://github.com/D-X-Y/NATS-Bench/issues/2
    api = self.prepare_fake_tss()
    data = api.query_by_index(284, dataname='cifar10', hp=200)
    for xkey, xvalue in data.items():
      print('{:} : {:}'.format(xkey, xvalue))
    xinfo = data[777].get_train()
    print(xinfo)
    print(data[777].train_acc1es)

    info_012_epochs = api.get_more_info(284, 'cifar10', hp= 12)
    print('Train accuracy for  12 epochs is {:}'.format(info_012_epochs['train-accuracy']))
    info_200_epochs = api.get_more_info(284, 'cifar10', hp=200)
    print('Train accuracy for 200 epochs is {:}'.format(info_200_epochs['train-accuracy']))
 

def _test_nats_bench(benchmark_dir, is_tss, fake_random, verbose=False):
  """The main test entry for NATS-Bench."""
  if is_tss:
    api = NATStopology(benchmark_dir, True, verbose)
  else:
    api = NATSsize(benchmark_dir, True, verbose)

  if fake_random:
    test_indexes = [0, 11, 284]
  else:
    test_indexes = [random.randint(0, len(api) - 1) for _ in range(10)]

  key2dataset = {'cifar10': 'CIFAR-10',
                 'cifar100': 'CIFAR-100',
                 'ImageNet16-120': 'ImageNet16-120'}

  for index in test_indexes:
    print('\n\nEvaluate the {:5d}-th architecture.'.format(index))

    for key, dataset in key2dataset.items():
      # Query the loss / accuracy / time for the `index`-th candidate
      #   architecture on CIFAR-10
      # info is a dict, where you can easily figure out the meaning by key
      info = api.get_more_info(index, key)
      print('  -->> The performance on {:}: {:}'.format(dataset, info))

      # Query the flops, params, latency. info is a dict.
      info = api.get_cost_info(index, key)
      print('  -->> The cost info on {:}: {:}'.format(dataset, info))

      # Simulate the training of the `index`-th candidate:
      validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(
          index, dataset=key, hp='12')
      print('  -->> The validation accuracy={:}, latency={:}, '
            'the current time cost={:} s, accumulated time cost={:} s'
            .format(validation_accuracy, latency, time_cost,
                    current_total_time_cost))

      # Print the configuration of the `index`-th architecture on CIFAR-10
      config = api.get_net_config(index, key)
      print('  -->> The configuration on {:} is {:}'.format(dataset, config))

    # Show the information of the `index`-th architecture
    api.show(index)

  with pytest.raises(ValueError):
    api.get_more_info(100000, 'cifar10')
