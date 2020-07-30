#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07 #
#####################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size
#####################################################
#
#
from .api_utils import ArchResults, ResultsCount
from .api_topology import NATStopology
from .api_size import NATSsize

NATS_BENCH_API_VERSIONs = ['v1.0']    # [2020.07.30]


def version():
  return NATS_BENCH_API_VERSIONs[-1]


def create(file_path_or_dict, search_space, verbose=True):
  if search_space in ['tss', 'topology']:
    return NATStopology(file_path_or_dict, verbose)
  elif search_space in ['sss', 'size']:
    return NATSsize(file_path_or_dict, verbose)
  else:
    raise ValueError('invalid search space : {:}'.format(search_space))
