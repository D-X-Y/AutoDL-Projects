###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import tensorflow as tf
import numpy as np
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import SearchCell


def sample_gumbel(shape, eps=1e-20):
  U = tf.random.uniform(shape, minval=0, maxval=1)
  return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax(logits, temperature):
  gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
  y = tf.nn.softmax(gumbel_softmax_sample / temperature)
  return y


class TinyNetworkGDAS(tf.keras.Model):

  def __init__(self, C, N, max_nodes, num_classes, search_space, affine):
    super(TinyNetworkGDAS, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self.stem = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(16, 3, 1, padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization()], name='stem')
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      cell_prefix = 'cell-{:03d}'.format(index)
      #with tf.name_scope(cell_prefix) as scope:
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      C_prev = cell.out_dim
      setattr(self, cell_prefix, cell)
    self.num_layers = len(layer_reductions)
    self.op_names   = deepcopy( search_space )
    self.edge2index = edge2index
    self.num_edge   = num_edge
    self.lastact    = tf.keras.Sequential([
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.GlobalAvgPool2D(),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(num_classes, activation='softmax')], name='lastact')
    #self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )
    arch_init = tf.random_normal_initializer(mean=0, stddev=0.001)
    self.arch_parameters = tf.Variable(initial_value=arch_init(shape=(num_edge, len(search_space)), dtype='float32'), trainable=True, name='arch-encoding')

  def get_alphas(self):
    xlist = self.trainable_variables
    return [x for x in xlist if 'arch-encoding' in x.name]

  def get_weights(self):
    xlist = self.trainable_variables
    return [x for x in xlist if 'arch-encoding' not in x.name]

  def get_np_alphas(self):
    arch_nps = self.arch_parameters.numpy()
    arch_ops = np.exp(arch_nps) / np.sum(np.exp(arch_nps), axis=-1, keepdims=True)
    return arch_ops

  def genotype(self):
    genotypes, arch_nps = [], self.arch_parameters.numpy()
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights = arch_nps[ self.edge2index[node_str] ]
        op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return genotypes

  # 
  def call(self, inputs, tau, training):
    weightss = tf.cond(tau < 0, lambda: tf.nn.softmax(self.arch_parameters, axis=1),
                                lambda: gumbel_softmax(tf.math.log_softmax(self.arch_parameters, axis=1), tau))
    feature = self.stem(inputs, training)
    for idx in range(self.num_layers):
      cell = getattr(self, 'cell-{:03d}'.format(idx))
      if isinstance(cell, SearchCell):
        feature = cell.call(feature, weightss, training)
      else:
        feature = cell(feature, training)
    logits = self.lastact(feature, training)
    return logits
