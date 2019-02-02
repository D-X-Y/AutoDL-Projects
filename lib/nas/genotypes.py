from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('sep_conv_5x5', 0, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('avg_pool_3x3', 1, 1.0),
    ('skip_connect', 0, 1.0),
    ('avg_pool_3x3', 0, 1.0),
    ('avg_pool_3x3', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('skip_connect', 1, 1.0),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1, 1.0),
    ('sep_conv_7x7', 0, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('sep_conv_7x7', 0, 1.0),
    ('avg_pool_3x3', 1, 1.0),
    ('sep_conv_5x5', 0, 1.0),
    ('skip_connect', 3, 1.0),
    ('avg_pool_3x3', 2, 1.0),
    ('sep_conv_3x3', 2, 1.0),
    ('max_pool_3x3', 1, 1.0),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('sep_conv_5x5', 2, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('avg_pool_3x3', 3, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('skip_connect', 1, 1.0),
    ('skip_connect', 0, 1.0),
    ('avg_pool_3x3', 1, 1.0),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('max_pool_3x3', 0, 1.0),
    ('sep_conv_7x7', 2, 1.0),
    ('sep_conv_7x7', 0, 1.0),
    ('avg_pool_3x3', 1, 1.0),
    ('max_pool_3x3', 0, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('conv_7x1_1x7', 0, 1.0),
    ('sep_conv_3x3', 5, 1.0),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(
  normal=[
    ('sep_conv_3x3', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('skip_connect', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('skip_connect', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('skip_connect', 2, 1.0)],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    ('max_pool_3x3', 0, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('skip_connect', 2, 1.0),
    ('max_pool_3x3', 0, 1.0),
    ('max_pool_3x3', 0, 1.0),
    ('skip_connect', 2, 1.0),
    ('skip_connect', 2, 1.0),
    ('avg_pool_3x3', 0, 1.0)],
  reduce_concat=[2, 3, 4, 5]
)

DARTS_V2 = Genotype(
  normal=[
    ('sep_conv_3x3', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('skip_connect', 0, 1.0),
    ('skip_connect', 0, 1.0),
    ('dil_conv_3x3', 2, 1.0)],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    ('max_pool_3x3', 0, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('skip_connect', 2, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('max_pool_3x3', 0, 1.0),
    ('skip_connect', 2, 1.0),
    ('skip_connect', 2, 1.0),
    ('max_pool_3x3', 1, 1.0)],
  reduce_concat=[2, 3, 4, 5]
)

PNASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 0, 1.0),
    ('max_pool_3x3', 0, 1.0),
    ('sep_conv_7x7', 1, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('sep_conv_5x5', 1, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('sep_conv_3x3', 4, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('skip_connect', 1, 1.0),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 0, 1.0),
    ('max_pool_3x3', 0, 1.0),
    ('sep_conv_7x7', 1, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('sep_conv_5x5', 1, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('sep_conv_3x3', 4, 1.0),
    ('max_pool_3x3', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('skip_connect', 1, 1.0),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)

# https://arxiv.org/pdf/1802.03268.pdf
ENASNet = Genotype(
  normal = [
    ('sep_conv_3x3', 1, 1.0),
    ('skip_connect', 1, 1.0),
    ('sep_conv_5x5', 1, 1.0),
    ('skip_connect', 0, 1.0),
    ('avg_pool_3x3', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0),
    ('sep_conv_3x3', 0, 1.0),
    ('avg_pool_3x3', 1, 1.0),
    ('sep_conv_5x5', 1, 1.0),
    ('avg_pool_3x3', 0, 1.0),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 0, 1.0),
    ('sep_conv_3x3', 1, 1.0), # 2
    ('sep_conv_3x3', 1, 1.0),
    ('avg_pool_3x3', 1, 1.0), # 3
    ('sep_conv_3x3', 1, 1.0),
    ('avg_pool_3x3', 1, 1.0), # 4
    ('avg_pool_3x3', 1, 1.0),
    ('sep_conv_5x5', 4, 1.0), # 5
    ('sep_conv_3x3', 5, 1.0),
    ('sep_conv_5x5', 0, 1.0),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)

DARTS = DARTS_V2

# Search by normal and reduce
GDAS_V1 = Genotype(
  normal=[('skip_connect', 0, 0.13017432391643524), ('skip_connect', 1, 0.12947972118854523), ('skip_connect', 0, 0.13062666356563568), ('sep_conv_5x5', 2, 0.12980839610099792), ('sep_conv_3x3', 3, 0.12923765182495117), ('skip_connect', 0, 0.12901571393013), ('sep_conv_5x5', 4, 0.12938997149467468), ('sep_conv_3x3', 3, 0.1289220005273819)],
  normal_concat=range(2, 6),
  reduce=[('sep_conv_5x5', 0, 0.12862831354141235), ('sep_conv_3x3', 1, 0.12783904373645782), ('sep_conv_5x5', 2, 0.12725995481014252), ('sep_conv_5x5', 1, 0.12705285847187042), ('dil_conv_5x5', 2, 0.12797553837299347), ('sep_conv_3x3', 1, 0.12737272679805756), ('sep_conv_5x5', 0, 0.12833961844444275), ('sep_conv_5x5', 1, 0.12758426368236542)],
  reduce_concat=range(2, 6)
)

# Search by normal and fixing reduction
GDAS_F1 = Genotype(
  normal=[('skip_connect', 0, 0.16), ('skip_connect', 1, 0.13), ('skip_connect', 0, 0.17), ('sep_conv_3x3', 2, 0.15), ('skip_connect', 0, 0.17), ('sep_conv_3x3', 2, 0.15), ('skip_connect', 0, 0.16), ('sep_conv_3x3', 2, 0.15)],
  normal_concat=[2, 3, 4, 5],
  reduce=None,
  reduce_concat=[2, 3, 4, 5],
)

# Combine DMS_V1 and DMS_F1
GDAS_GF = Genotype(
  normal=[('skip_connect', 0, 0.13017432391643524), ('skip_connect', 1, 0.12947972118854523), ('skip_connect', 0, 0.13062666356563568), ('sep_conv_5x5', 2, 0.12980839610099792), ('sep_conv_3x3', 3, 0.12923765182495117), ('skip_connect', 0, 0.12901571393013), ('sep_conv_5x5', 4, 0.12938997149467468), ('sep_conv_3x3', 3, 0.1289220005273819)],
  normal_concat=range(2, 6),
  reduce=None,
  reduce_concat=range(2, 6)
)
GDAS_FG = Genotype(
  normal=[('skip_connect', 0, 0.16), ('skip_connect', 1, 0.13), ('skip_connect', 0, 0.17), ('sep_conv_3x3', 2, 0.15), ('skip_connect', 0, 0.17), ('sep_conv_3x3', 2, 0.15), ('skip_connect', 0, 0.16), ('sep_conv_3x3', 2, 0.15)],
  normal_concat=range(2, 6),
  reduce=[('sep_conv_5x5', 0, 0.12862831354141235), ('sep_conv_3x3', 1, 0.12783904373645782), ('sep_conv_5x5', 2, 0.12725995481014252), ('sep_conv_5x5', 1, 0.12705285847187042), ('dil_conv_5x5', 2, 0.12797553837299347), ('sep_conv_3x3', 1, 0.12737272679805756), ('sep_conv_5x5', 0, 0.12833961844444275), ('sep_conv_5x5', 1, 0.12758426368236542)],
  reduce_concat=range(2, 6)
)

model_types = {'DARTS_V1': DARTS_V1,
               'DARTS_V2': DARTS_V2,
               'NASNet'  : NASNet,
               'PNASNet' : PNASNet, 
               'AmoebaNet': AmoebaNet,
               'ENASNet' : ENASNet,
               'GDAS_V1' : GDAS_V1,
               'GDAS_F1' : GDAS_F1,
               'GDAS_GF' : GDAS_GF,
               'GDAS_FG' : GDAS_FG}
