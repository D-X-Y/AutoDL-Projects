#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import spaces
from .super_module import IntSpaceType
from .super_module import BoolSpaceType
from .super_module import SuperModule
from .super_linear import SuperMLPv2
from .super_norm import SuperLayerNorm1D
from .super_attention import SuperAttention


class SuperTransformerEncoderLayer(SuperModule):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This is a super model for TransformerEncoderLayer that can support search for the transformer encoder layer.

    Reference:
      - Paper: Attention Is All You Need, NeurIPS 2017
      - PyTorch Implementation: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

    Details:
      MHA -> residual -> norm -> MLP -> residual -> norm
    """

    def __init__(
        self,
        input_dim: IntSpaceType,
        output_dim: IntSpaceType,
        num_heads: IntSpaceType,
        qkv_bias: BoolSpaceType = False,
        mlp_hidden_multiplier: IntSpaceType = 4,
        drop: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ):
        super(SuperTransformerEncoderLayer, self).__init__()
        self.mha = SuperAttention(
            input_dim,
            input_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.drop1 = nn.Dropout(drop or 0.0)
        self.norm1 = SuperLayerNorm1D(input_dim)
        self.mlp = SuperMLPv2(
            input_dim,
            hidden_multiplier=mlp_hidden_multiplier,
            out_features=output_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop2 = nn.Dropout(drop or 0.0)
        self.norm2 = SuperLayerNorm1D(output_dim)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        xdict = dict(
            mha=self.mha.abstract_search_space,
            norm1=self.norm1.abstract_search_space,
            mlp=self.mlp.abstract_search_space,
            norm2=self.norm2.abstract_search_space,
        )
        for key, space in xdict.items():
            if not spaces.is_determined(space):
                root_node.append(key, space)
        return root_node

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperTransformerEncoderLayer, self).apply_candidate(abstract_child)
        valid_keys = ["mha", "norm1", "mlp", "norm2"]
        for key in valid_keys:
            if key in abstract_child:
                getattr(self, key).apply_candidate(abstract_child[key])

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        # multi-head attention
        x = self.mha(input)
        x = x + self.drop1(x)
        x = self.norm1(x)
        # feed-forward layer
        x = self.mlp(x)
        x = x + self.drop2(x)
        x = self.norm2(x)
        return x
