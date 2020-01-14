##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################
# I write this package to make AutoDL-Projects to be compatible with the old GDAS projects.
# Ideally, this package will be merged into lib/models/cell_infers in future.
# Currently, this package is used to reproduce the results in GDAS (Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019).
##################################################

import torch

def obtain_nas_infer_model(config):
  if config.arch == 'dxys':
    from .DXYs import CifarNet, ImageNet, Networks
    genotype = Networks[config.genotype]
    if config.dataset == 'cifar':
      return CifarNet(config.ichannel, config.layers, config.stem_multi, config.auxiliary, genotype, config.class_num)
    elif config.dataset == 'imagenet':
      return ImageNet(config.ichannel, config.layers, config.auxiliary, genotype, config.class_num)
    else: raise ValueError('invalid dataset : {:}'.format(config.dataset))
  else:
    raise ValueError('invalid nas arch type : {:}'.format(config.arch))
