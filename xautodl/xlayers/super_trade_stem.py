#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xautodl import spaces
from .super_linear import SuperLinear
from .super_module import SuperModule
from .super_module import IntSpaceType


class SuperAlphaEBDv1(SuperModule):
    """A simple layer to convert the raw trading data from 1-D to 2-D data and apply an FC layer."""

    def __init__(self, d_feat: int, embed_dim: IntSpaceType):
        super(SuperAlphaEBDv1, self).__init__()
        self._d_feat = d_feat
        self._embed_dim = embed_dim
        self.proj = SuperLinear(d_feat, embed_dim)

    @property
    def embed_dim(self):
        return spaces.get_max(self._embed_dim)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        space = self.proj.abstract_search_space
        if not spaces.is_determined(space):
            root_node.append("proj", space)
        if not spaces.is_determined(self._embed_dim):
            root_node.append("_embed_dim", self._embed_dim.abstract(reuse_last=True))
        return root_node

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperAlphaEBDv1, self).apply_candidate(abstract_child)
        if "proj" in abstract_child:
            self.proj.apply_candidate(abstract_child["proj"])

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        x = input.reshape(len(input), self._d_feat, -1)  # [N, F*T] -> [N, F, T]
        x = x.permute(0, 2, 1)  # [N, F, T] -> [N, T, F]
        if not spaces.is_determined(self._embed_dim):
            embed_dim = self.abstract_child["_embed_dim"].value
        else:
            embed_dim = spaces.get_determined_value(self._embed_dim)
        out = self.proj(x) * math.sqrt(embed_dim)
        return out

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        x = input.reshape(len(input), self._d_feat, -1)  # [N, F*T] -> [N, F, T]
        x = x.permute(0, 2, 1)  # [N, F, T] -> [N, T, F]
        out = self.proj(x) * math.sqrt(self.embed_dim)
        return out
