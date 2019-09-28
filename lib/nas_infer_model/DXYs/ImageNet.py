import torch
import torch.nn as nn
from .construct_utils import drop_path
from .base_cells import InferCell
from .head_utils import ImageNetHEAD, AuxiliaryHeadImageNet


class NetworkImageNet(nn.Module):

  def __init__(self, C, N, auxiliary, genotype, num_classes):
    super(NetworkImageNet, self).__init__()
    self._C          = C
    self._layerN     = N
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr, reduction_prev = C, C, C, True

    self.cells = nn.ModuleList()
    self.auxiliary_index = None
    for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      cell = InferCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell._multiplier * C_curr
      if reduction and C_curr == C*4:
        C_to_auxiliary = C_prev
        self.auxiliary_index = i
  
    self._NNN = len(self.cells)
    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    else:
      self.auxiliary_head = None
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier     = nn.Linear(C_prev, num_classes)
    self.drop_path_prob = -1

  def update_drop_path(self, drop_path_prob):
    self.drop_path_prob = drop_path_prob

  def extra_repr(self):
    return ('{name}(C={_C}, N=[{_layerN}, {_NNN}], aux-index={auxiliary_index}, drop-path={drop_path_prob})'.format(name=self.__class__.__name__, **self.__dict__))

  def get_message(self):
    return self.extra_repr()

  def auxiliary_param(self):
    if self.auxiliary_head is None: return []
    else: return list( self.auxiliary_head.parameters() )

  def forward(self, inputs):
    s0 = self.stem0(inputs)
    s1 = self.stem1(s0)
    logits_aux = None
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == self.auxiliary_index and self.auxiliary_head and self.training:
        logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))

    if logits_aux is None: return out, logits
    else                 : return out, [logits, logits_aux]
