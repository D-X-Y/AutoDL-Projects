##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018
NASNet = Genotype(
  normal = [
    (('sep_conv_5x5', 1), ('sep_conv_3x3', 0)),
    (('sep_conv_5x5', 0), ('sep_conv_3x3', 0)),
    (('avg_pool_3x3', 1), ('skip_connect', 0)),
    (('avg_pool_3x3', 0), ('avg_pool_3x3', 0)),
    (('sep_conv_3x3', 1), ('skip_connect', 1)),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    (('sep_conv_5x5', 1), ('sep_conv_7x7', 0)),
    (('max_pool_3x3', 1), ('sep_conv_7x7', 0)),
    (('avg_pool_3x3', 1), ('sep_conv_5x5', 0)),
    (('skip_connect', 3), ('avg_pool_3x3', 2)),
    (('sep_conv_3x3', 2), ('max_pool_3x3', 1)),
  ],
  reduce_concat = [4, 5, 6],
)


# Progressive Neural Architecture Search, ECCV 2018
PNASNet = Genotype(
  normal = [
    (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
    (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
    (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
    (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
    (('sep_conv_3x3', 0), ('skip_connect', 1)),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
    (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
    (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
    (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
    (('sep_conv_3x3', 0), ('skip_connect', 1)),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)


# Regularized Evolution for Image Classifier Architecture Search, AAAI 2019
AmoebaNet = Genotype(
  normal = [
    (('avg_pool_3x3', 0), ('max_pool_3x3', 1)),
    (('sep_conv_3x3', 0), ('sep_conv_5x5', 2)),
    (('sep_conv_3x3', 0), ('avg_pool_3x3', 3)),
    (('sep_conv_3x3', 1), ('skip_connect', 1)),
    (('skip_connect', 0), ('avg_pool_3x3', 1)),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    (('avg_pool_3x3', 0), ('sep_conv_3x3', 1)),
    (('max_pool_3x3', 0), ('sep_conv_7x7', 2)),
    (('sep_conv_7x7', 0), ('avg_pool_3x3', 1)),
    (('max_pool_3x3', 0), ('max_pool_3x3', 1)),
    (('conv_7x1_1x7', 0), ('sep_conv_3x3', 5)),
  ],
  reduce_concat = [3, 4, 6]
)


# Efficient Neural Architecture Search via Parameter Sharing, ICML 2018
ENASNet = Genotype(
  normal = [
    (('sep_conv_3x3', 1), ('skip_connect', 1)),
    (('sep_conv_5x5', 1), ('skip_connect', 0)),
    (('avg_pool_3x3', 0), ('sep_conv_3x3', 1)),
    (('sep_conv_3x3', 0), ('avg_pool_3x3', 1)),
    (('sep_conv_5x5', 1), ('avg_pool_3x3', 0)),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    (('sep_conv_5x5', 0), ('sep_conv_3x3', 1)), # 2
    (('sep_conv_3x3', 1), ('avg_pool_3x3', 1)), # 3
    (('sep_conv_3x3', 1), ('avg_pool_3x3', 1)), # 4
    (('avg_pool_3x3', 1), ('sep_conv_5x5', 4)), # 5
    (('sep_conv_3x3', 5), ('sep_conv_5x5', 0)),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)


# DARTS: Differentiable Architecture Search, ICLR 2019
DARTS_V1 = Genotype(
  normal=[
    (('sep_conv_3x3', 1), ('sep_conv_3x3', 0)), # step 1
    (('skip_connect', 0), ('sep_conv_3x3', 1)), # step 2
    (('skip_connect', 0), ('sep_conv_3x3', 1)), # step 3
    (('sep_conv_3x3', 0), ('skip_connect', 2))  # step 4
  ],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('max_pool_3x3', 0), ('max_pool_3x3', 1)), # step 1
    (('skip_connect', 2), ('max_pool_3x3', 0)), # step 2
    (('max_pool_3x3', 0), ('skip_connect', 2)), # step 3
    (('skip_connect', 2), ('avg_pool_3x3', 0))  # step 4
  ],
  reduce_concat=[2, 3, 4, 5],
)


# DARTS: Differentiable Architecture Search, ICLR 2019
DARTS_V2 = Genotype(
  normal=[
    (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), # step 1
    (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), # step 2
    (('sep_conv_3x3', 1), ('skip_connect', 0)), # step 3
    (('skip_connect', 0), ('dil_conv_3x3', 2))  # step 4
  ],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('max_pool_3x3', 0), ('max_pool_3x3', 1)), # step 1
    (('skip_connect', 2), ('max_pool_3x3', 1)), # step 2
    (('max_pool_3x3', 0), ('skip_connect', 2)), # step 3
    (('skip_connect', 2), ('max_pool_3x3', 1))  # step 4
  ],
  reduce_concat=[2, 3, 4, 5],
)



# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019
SETN = Genotype(
  normal=[
    (('skip_connect', 0), ('sep_conv_5x5', 1)),
    (('sep_conv_5x5', 0), ('sep_conv_3x3', 1)),
    (('sep_conv_5x5', 1), ('sep_conv_5x5', 3)),
    (('max_pool_3x3', 1), ('conv_3x1_1x3', 4))],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('sep_conv_3x3', 0), ('sep_conv_5x5', 1)),
    (('avg_pool_3x3', 0), ('sep_conv_5x5', 1)),
    (('avg_pool_3x3', 0), ('sep_conv_5x5', 1)),
    (('avg_pool_3x3', 0), ('skip_connect', 1))],
  reduce_concat=[2, 3, 4, 5],
)


# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019
GDAS_V1 = Genotype(
  normal=[
    (('skip_connect', 0), ('skip_connect', 1)),
    (('skip_connect', 0), ('sep_conv_5x5', 2)),
    (('sep_conv_3x3', 3), ('skip_connect', 0)),
    (('sep_conv_5x5', 4), ('sep_conv_3x3', 3))],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('sep_conv_5x5', 0), ('sep_conv_3x3', 1)), 
    (('sep_conv_5x5', 2), ('sep_conv_5x5', 1)),
    (('dil_conv_5x5', 2), ('sep_conv_3x3', 1)),
    (('sep_conv_5x5', 0), ('sep_conv_5x5', 1))],
  reduce_concat=[2, 3, 4, 5],
)


Networks = {'DARTS_V1' : DARTS_V1,
            'DARTS_V2' : DARTS_V2,
            'DARTS'    : DARTS_V2,
            'NASNet'   : NASNet,
            'ENASNet'  : ENASNet,
            'AmoebaNet': AmoebaNet,
            'GDAS_V1'  : GDAS_V1,
            'PNASNet'  : PNASNet,
            'SETN'     : SETN,
           }
