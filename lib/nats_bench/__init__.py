##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 ##########################
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
# The official Application Programming Interface (API) for NATS-Bench.       #
##############################################################################
from .api_utils import pickle_save, pickle_load
from .api_utils import ArchResults, ResultsCount
from .api_topology import NATStopology
from .api_size import NATSsize


NATS_BENCH_API_VERSIONs = ['v1.0']    # [2020.08.31]


def version():
  return NATS_BENCH_API_VERSIONs[-1]


def create(file_path_or_dict, search_space, fast_mode=False, verbose=True):
  """Create the instead for NATS API.

  Args:
    file_path_or_dict: None or a file path or a directory path.
    search_space: This is a string indicates the search space in NATS-Bench.
    fast_mode: If True, we will not load all the data at initialization, instead, the data for each candidate architecture will be loaded when quering it;
               If False, we will load all the data during initialization.
    verbose: This is a flag to indicate whether log additional information.
  """
  if search_space in ['tss', 'topology']:
    return NATStopology(file_path_or_dict, fast_mode, verbose)
  elif search_space in ['sss', 'size']:
    return NATSsize(file_path_or_dict, fast_mode, verbose)
  else:
    raise ValueError('invalid search space : {:}'.format(search_space))
