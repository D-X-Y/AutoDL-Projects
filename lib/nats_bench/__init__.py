##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 ##########################
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
"""The official Application Programming Interface (API) for NATS-Bench."""
from nats_bench.api_size import NATSsize
from nats_bench.api_topology import NATStopology
from nats_bench.api_utils import ArchResults
from nats_bench.api_utils import pickle_load
from nats_bench.api_utils import pickle_save
from nats_bench.api_utils import ResultsCount


NATS_BENCH_API_VERSIONs = ['v1.0']    # [2020.08.31]
NATS_BENCH_SSS_NAMEs = ('sss', 'size')
NATS_BENCH_TSS_NAMEs = ('tss', 'topology')


def version():
  return NATS_BENCH_API_VERSIONs[-1]


def create(file_path_or_dict, search_space, fast_mode=False, verbose=True):
  """Create the instead for NATS API.

  Args:
    file_path_or_dict: None or a file path or a directory path.
    search_space: This is a string indicates the search space in NATS-Bench.
    fast_mode: If True, we will not load all the data at initialization,
      instead, the data for each candidate architecture will be loaded when
      quering it; If False, we will load all the data during initialization.
    verbose: This is a flag to indicate whether log additional information.

  Raises:
    ValueError: If not find the matched serach space description.

  Returns:
    The created NATS-Bench API.
  """
  if search_space in NATS_BENCH_TSS_NAMEs:
    return NATStopology(file_path_or_dict, fast_mode, verbose)
  elif search_space in NATS_BENCH_SSS_NAMEs:
    return NATSsize(file_path_or_dict, fast_mode, verbose)
  else:
    raise ValueError('invalid search space : {:}'.format(search_space))


def search_space_info(main_tag, aux_tag):
  """Obtain the search space information."""
  nats_sss = dict(candidates=[8, 16, 24, 32, 40, 48, 56, 64],
                  num_layers=5)
  nats_tss = dict(op_names=['none', 'skip_connect',
                            'nor_conv_1x1', 'nor_conv_3x3',
                            'avg_pool_3x3'],
                  num_nodes=4)
  if main_tag == 'nats-bench':
    if aux_tag in NATS_BENCH_SSS_NAMEs:
      return nats_sss
    elif aux_tag in NATS_BENCH_TSS_NAMEs:
      return nats_tss
    else:
      raise ValueError('Unknown auxiliary tag: {:}'.format(aux_tag))
  elif main_tag == 'nas-bench-201':
    if aux_tag is not None:
      raise ValueError('For NAS-Bench-201, the auxiliary tag should be None.')
    return nats_tss
  else:
    raise ValueError('Unknown main tag: {:}'.format(main_tag))
