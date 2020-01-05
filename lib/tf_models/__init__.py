##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
from os import path as osp

__all__ = ['get_cell_based_tiny_net', 'get_search_spaces']


# the cell-based NAS models
def get_cell_based_tiny_net(config):
  group_names = ['GDAS']
  if config.name in group_names:
    from .cell_searchs import nas_super_nets
    from .cell_operations import SearchSpaceNames
    if isinstance(config.space, str): search_space = SearchSpaceNames[config.space]
    else: search_space = config.space
    return nas_super_nets[config.name](
                  config.C, config.N, config.max_nodes,
                  config.num_classes, search_space, config.affine)
  else:
    raise ValueError('invalid network name : {:}'.format(config.name))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name):
  if xtype == 'cell':
    from .cell_operations import SearchSpaceNames
    assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
    return SearchSpaceNames[name]
  else:
    raise ValueError('invalid search-space type is {:}'.format(xtype))
