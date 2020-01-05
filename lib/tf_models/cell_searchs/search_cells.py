##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random
import tensorflow as tf
from copy import deepcopy
from ..cell_operations import OPS


class SearchCell(tf.keras.layers.Layer):

  def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False):
    super(SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.max_nodes = max_nodes
    self.in_dim    = C_in
    self.out_dim   = C_out
    self.edge_keys = []
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if j == 0:
          xlists = [OPS[op_name](C_in , C_out, stride, affine) for op_name in op_names]
        else:
          xlists = [OPS[op_name](C_in , C_out,      1, affine) for op_name in op_names]
        for k, op in enumerate(xlists):
          setattr(self, '{:}.{:}'.format(node_str, k), op)
        self.edge_keys.append( node_str )
    self.edge_keys  = sorted(self.edge_keys)
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edge_keys)

  def call(self, inputs, weightss, training):
    w_lst = tf.split(weightss, self.num_edges, 0)
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        edge_idx = self.edge2index[node_str]
        op_outps = []
        for k, op_name in enumerate(self.op_names):
          op = getattr(self, '{:}.{:}'.format(node_str, k))
          op_outps.append( op(nodes[j], training) )
        stack_op_outs = tf.stack(op_outps, axis=-1)
        weighted_sums = tf.math.multiply(stack_op_outs, w_lst[edge_idx])
        inter_nodes.append( tf.math.reduce_sum(weighted_sums, axis=-1) )
      nodes.append( tf.math.add_n(inter_nodes) )
    return nodes[-1]
