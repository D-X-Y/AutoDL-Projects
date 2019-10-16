import math, torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .operations import OPS, ReLUConvBN


class SearchCell(nn.Module):

  def __init__(self, C_in, C_out, stride, max_nodes, op_names):
    super(SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.edges     = nn.ModuleDict()
    self.max_nodes = max_nodes
    self.in_dim    = C_in
    self.out_dim   = C_out
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if j == 0:
          xlists = [OPS[op_name](C_in , C_out, stride) for op_name in op_names]
        else:
          xlists = [OPS[op_name](C_in , C_out,      1) for op_name in op_names]
        self.edges[ node_str ] = nn.ModuleList( xlists )
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)

  def extra_repr(self):
    string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    return string

  def forward(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # GDAS
  def forward_acc(self, inputs, weightss, indexess):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        indexes  = indexess[ self.edge2index[node_str] ].item()
        import pdb; pdb.set_trace() # to-do
        #inter_nodes.append( self.edges[node_str][indexes](nodes[j]) * weights[indexes] )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # joint
  def forward_joint(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
        inter_nodes.append( aggregation )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # uniform random sampling per iteration
  def forward_urs(self, inputs):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      while True: # to avoid select zero for all ops
        sops, has_non_zero = [], False
        for j in range(i):
          node_str   = '{:}<-{:}'.format(i, j)
          candidates = self.edges[node_str]
          select_op  = random.choice(candidates)
          sops.append( select_op )
          if not hasattr(select_op, 'is_zero') or select_op.is_zero == False: has_non_zero=True
        if has_non_zero: break
      inter_nodes = []
      for j, select_op in enumerate(sops):
        inter_nodes.append( select_op(nodes[j]) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # select the argmax
  def forward_select(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( self.edges[node_str][ weights.argmax().item() ]( nodes[j] ) )
        #inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # select the argmax
  def forward_dynamic(self, inputs, structure):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      cur_op_node = structure.nodes[i-1]
      inter_nodes = []
      for op_name, j in cur_op_node:
        node_str = '{:}<-{:}'.format(i, j)
        op_index = self.op_names.index( op_name )
        inter_nodes.append( self.edges[node_str][op_index]( nodes[j] ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]


class InferCell(nn.Module):

  def __init__(self, genotype, C_in, C_out, stride):
    super(InferCell, self).__init__()

    self.layers  = nn.ModuleList()
    self.node_IN = []
    self.node_IX = []
    self.genotype = deepcopy(genotype)
    for i in range(1, len(genotype)):
      node_info = genotype[i-1]
      cur_index = []
      cur_innod = []
      for (op_name, op_in) in node_info:
        if op_in == 0:
          layer = OPS[op_name](C_in , C_out, stride)
        else:
          layer = OPS[op_name](C_out, C_out,      1)
        cur_index.append( len(self.layers) )
        cur_innod.append( op_in )
        self.layers.append( layer )
      self.node_IX.append( cur_index )
      self.node_IN.append( cur_innod )
    self.nodes   = len(genotype)
    self.in_dim  = C_in
    self.out_dim = C_out

  def extra_repr(self):
    string = 'info :: nodes={nodes}, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    laystr = []
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      y = ['I{:}-L{:}'.format(_ii, _il) for _il, _ii in zip(node_layers, node_innods)]
      x = '{:}<-({:})'.format(i+1, ','.join(y))
      laystr.append( x )
    return string + ', [{:}]'.format( ' | '.join(laystr) ) + ', {:}'.format(self.genotype.tostr())

  def forward(self, inputs):
    nodes = [inputs]
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      node_feature = sum( self.layers[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods) )
      nodes.append( node_feature )
    return nodes[-1]



class ResNetBasicblock(nn.Module):

  def __init__(self, inplanes, planes, stride):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1)
    self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1)
    if stride == 2:
      self.downsample = nn.Sequential(
                           nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
    elif inplanes != planes:
      self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1)
    else:
      self.downsample = None
    self.in_dim  = inplanes
    self.out_dim = planes
    self.stride  = stride
    self.num_conv = 2

  def extra_repr(self):
    string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
    return string

  def forward(self, inputs):

    basicblock = self.conv_a(inputs)
    basicblock = self.conv_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(inputs)
    else:
      residual = inputs
    return residual + basicblock
