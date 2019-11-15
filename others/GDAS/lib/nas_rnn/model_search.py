import copy, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from .genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from .basemodel import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)
    self.check_zero = False

  def set_check(self, check_zero):
    self.check_zero = check_zero

  def cell(self, x, h_prev, x_mask, h_mask, arch_probs):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    if self.check_zero:
      arch_probs_cpu = arch_probs.cpu().tolist()
    #arch_probs = F.softmax(self.weights, dim=-1)

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid)
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if name == 'none':
          continue
        fn = self._get_activation(name)
        unweighted = states + c * (fn(h) - states)
        if self.check_zero:
          INDEX, INDDX = [], []
          for jj in range(offset, offset+i+1):
            if arch_probs_cpu[jj][k] > 0:
              INDEX.append(jj)
              INDDX.append(jj-offset)
          if len(INDEX) == 0: continue
          s += torch.sum(arch_probs[INDEX, k].unsqueeze(-1).unsqueeze(-1) * unweighted[INDDX, :, :], dim=0)
        else:
          s += torch.sum(arch_probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
      s = self.bn(s)
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

  def __init__(self, *args):
    super(RNNModelSearch, self).__init__(*args)
    self._args = copy.deepcopy( args )

    k = sum(i for i in range(1, STEPS+1))
    self.arch_weights = nn.Parameter(torch.Tensor(k, len(PRIMITIVES)))
    nn.init.normal_(self.arch_weights, 0, 0.001)

  def base_parameters(self):
    lists  = list(self.lockdrop.parameters())
    lists += list(self.encoder.parameters())
    lists += list(self.rnns.parameters())
    lists += list(self.decoder.parameters())
    return lists

  def arch_parameters(self):
    return [self.arch_weights]

  def genotype(self):

    def _parse(probs):
      gene = []
      start = 0
      for i in range(STEPS):
        end = start + i + 1
        W = probs[start:end].copy()
        #j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
        j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) ))[0]
        k_best = None
        for k in range(len(W[j])):
          #if k != PRIMITIVES.index('none'):
          #  if k_best is None or W[j][k] > W[j][k_best]:
          #    k_best = k
          if k_best is None or W[j][k] > W[j][k_best]:
            k_best = k
        gene.append((PRIMITIVES[k_best], j))
        start = end
      return gene

    with torch.no_grad():
      gene = _parse(F.softmax(self.arch_weights, dim=-1).cpu().numpy())
    genotype = Genotype(recurrent=gene, concat=list(range(STEPS+1)[-CONCAT:]))
    return genotype
