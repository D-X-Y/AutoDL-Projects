import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .construct_utils import drop_path
from ..operations import OPS, Identity, FactorizedReduce, ReLUConvBN


class MixedOp(nn.Module):

  def __init__(self, C, stride, PRIMITIVES):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.name2idx = {}
    for idx, primitive in enumerate(PRIMITIVES):
      op = OPS[primitive](C, C, stride, False)
      self._ops.append(op)
      assert primitive not in self.name2idx, '{:} has already in'.format(primitive)
      self.name2idx[primitive] = idx

  def forward(self, x, weights, op_name):
    if op_name is None:
      if weights is None:
        return [op(x) for op in self._ops]
      else:
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    else:
      op_index = self.name2idx[op_name]
      return self._ops[op_index](x)



class SearchCell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, PRIMITIVES, use_residual):
    super(SearchCell, self).__init__()
    self.reduction  = reduction
    self.PRIMITIVES = deepcopy(PRIMITIVES)
  
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps        = steps
    self._multiplier   = multiplier
    self._use_residual = use_residual

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, self.PRIMITIVES)
        self._ops.append(op)

  def extra_repr(self):
    return ('{name}(residual={_use_residual}, steps={_steps}, multiplier={_multiplier})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, S0, S1, weights, connect, adjacency, drop_prob, modes):
    if modes[0] is None:
      if modes[1] == 'normal':
        output = self.__forwardBoth(S0, S1, weights, connect, adjacency, drop_prob)
      elif modes[1] == 'only_W':
        output = self.__forwardOnlyW(S0, S1, drop_prob)
    else:
      test_genotype = modes[0]
      if self.reduction: operations, concats = test_genotype.reduce, test_genotype.reduce_concat
      else             : operations, concats = test_genotype.normal, test_genotype.normal_concat
      s0, s1 = self.preprocess0(S0), self.preprocess1(S1)
      states, offset = [s0, s1], 0
      assert self._steps == len(operations), '{:} vs. {:}'.format(self._steps, len(operations))
      for i, (opA, opB) in enumerate(operations):
        A = self._ops[offset + opA[1]](states[opA[1]], None, opA[0])
        B = self._ops[offset + opB[1]](states[opB[1]], None, opB[0])
        state = A + B
        offset += len(states)
        states.append(state)
      output = torch.cat([states[i] for i in concats], dim=1)
    if self._use_residual and S1.size() == output.size():
      return S1 + output
    else: return output
  
  def __forwardBoth(self, S0, S1, weights, connect, adjacency, drop_prob):
    s0, s1 = self.preprocess0(S0), self.preprocess1(S1)
    states, offset = [s0, s1], 0
    for i in range(self._steps):
      clist = []
      for j, h in enumerate(states):
        x = self._ops[offset+j](h, weights[offset+j], None)
        if self.training and drop_prob > 0.:
          x = drop_path(x, math.pow(drop_prob, 1./len(states)))
        clist.append( x )
      connection = torch.mm(connect['{:}'.format(i)], adjacency[i]).squeeze(0)
      state = sum(w * node for w, node in zip(connection, clist))
      offset += len(states)
      states.append(state)
    return torch.cat(states[-self._multiplier:], dim=1)

  def __forwardOnlyW(self, S0, S1, drop_prob):
    s0, s1 = self.preprocess0(S0), self.preprocess1(S1)
    states, offset = [s0, s1], 0
    for i in range(self._steps):
      clist = []
      for j, h in enumerate(states):
        xs = self._ops[offset+j](h, None, None)
        clist += xs
      if self.training and drop_prob > 0.:
        xlist = [drop_path(x, math.pow(drop_prob, 1./len(states))) for x in clist]
      else: xlist = clist
      state = sum(xlist) * 2 / len(xlist)
      offset += len(states)
      states.append(state)
    return torch.cat(states[-self._multiplier:], dim=1)



class InferCell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(InferCell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev is None:
      self.preprocess0 = Identity()
    elif reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1   = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction: step_ops, concat = genotype.reduce, genotype.reduce_concat
    else        : step_ops, concat = genotype.normal, genotype.normal_concat
    self._steps        = len(step_ops)
    self._concat       = concat
    self._multiplier   = len(concat)
    self._ops          = nn.ModuleList()
    self._indices      = []
    for operations in step_ops:
      for name, index in operations:
        stride = 2 if reduction and index < 2 else 1
        if reduction_prev is None and index == 0:
          op = OPS[name](C_prev_prev, C, stride, True)
        else:
          op = OPS[name](C          , C, stride, True)
        self._ops.append( op )
        self._indices.append( index )

  def extra_repr(self):
    return ('{name}(steps={_steps}, concat={_concat})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, S0, S1, drop_prob):
    s0 = self.preprocess0(S0)
    s1 = self.preprocess1(S1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)

      state = h1 + h2
      states += [state]
    output = torch.cat([states[i] for i in self._concat], dim=1)
    return output
