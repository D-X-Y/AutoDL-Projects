##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 ##########################
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
"""This file is used to quickly test the API."""
import random

from nats_bench.api_size import NATSsize
from nats_bench.api_topology import NATStopology


def test_nats_bench_tss(benchmark_dir):
  return test_nats_bench(benchmark_dir, True)


def test_nats_bench_sss(benchmark_dir):
  return test_nats_bench(benchmark_dir, False)


def test_nats_bench(benchmark_dir, is_tss, verbose=False):
  if is_tss:
    api = NATStopology(benchmark_dir, True, verbose)
  else:
    api = NATSsize(benchmark_dir, True, verbose)

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
