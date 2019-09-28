##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
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
