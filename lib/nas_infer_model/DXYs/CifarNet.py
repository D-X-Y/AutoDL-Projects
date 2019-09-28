import torch
import torch.nn as nn
from .construct_utils import drop_path
from .head_utils      import CifarHEAD, AuxiliaryHeadCIFAR
from .base_cells      import InferCell


class NetworkCIFAR(nn.Module):

  def __init__(self, C, N, stem_multiplier, auxiliary, genotype, num_classes):
    super(NetworkCIFAR, self).__init__()
    self._C               = C
    self._layerN          = N
    self._stem_multiplier = stem_multiplier

    C_curr = self._stem_multiplier * C
    self.stem = CifarHEAD(C_curr)
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
    block_indexs     = [0    ] * N + [-1  ] + [1    ] * N + [-1  ] + [2    ] * N
    block2index      = {0:[], 1:[], 2:[]}

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    reduction_prev, spatial, dims = False, 1, []
    self.auxiliary_index = None
    self.auxiliary_head  = None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      cell = InferCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells.append( cell )
      C_prev_prev, C_prev = C_prev, cell._multiplier*C_curr
      if reduction and C_curr == C*4:
        if auxiliary:
          self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes)
          self.auxiliary_index = index

      if reduction: spatial *= 2
      dims.append( (C_prev, spatial) )
      
    self._Layer= len(self.cells)


    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.drop_path_prob = -1

  def update_drop_path(self, drop_path_prob):
    self.drop_path_prob = drop_path_prob

  def auxiliary_param(self):
    if self.auxiliary_head is None: return []
    else: return list( self.auxiliary_head.parameters() )

  def get_message(self):
    return self.extra_repr()

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, L={_Layer}, stem={_stem_multiplier}, drop-path={drop_path_prob})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):
    stem_feature, logits_aux = self.stem(inputs), None
    cell_results = [stem_feature, stem_feature]
    for i, cell in enumerate(self.cells):
      cell_feature = cell(cell_results[-2], cell_results[-1], self.drop_path_prob)
      cell_results.append( cell_feature )

      if self.auxiliary_index is not None and i == self.auxiliary_index and self.training:
        logits_aux = self.auxiliary_head( cell_results[-1] )
    out = self.global_pooling( cell_results[-1] )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    if logits_aux is None: return out, logits
    else                 : return out, [logits, logits_aux]
