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
from .super_module import LayerOrder
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
      the original post-norm version: MHA -> residual -> norm -> MLP -> residual -> norm
      the pre-norm version: norm -> MHA -> residual -> norm -> MLP -> residual
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
        order: LayerOrder = LayerOrder.PreNorm,
    ):
        super(SuperTransformerEncoderLayer, self).__init__()
        mha = SuperAttention(
            input_dim,
            input_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        drop1 = nn.Dropout(drop or 0.0)
        norm1 = SuperLayerNorm1D(input_dim)
        mlp = SuperMLPv2(
            input_dim,
            hidden_multiplier=mlp_hidden_multiplier,
            out_features=output_dim,
            act_layer=act_layer,
            drop=drop,
        )
        drop2 = nn.Dropout(drop or 0.0)
        norm2 = SuperLayerNorm1D(output_dim)
        if order is LayerOrder.PreNorm:
            self.norm1 = norm1
            self.mha = mha
            self.drop1 = drop1
            self.norm2 = norm2
            self.mlp = mlp
            self.drop2 = drop2
        elif order is LayerOrder.PostNoem:
            self.mha = mha
            self.drop1 = drop1
            self.norm1 = norm1
            self.mlp = mlp
            self.drop2 = drop2
            self.norm2 = norm2
        else:
            raise ValueError("Unknown order: {:}".format(order))

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
        if order is LayerOrder.PreNorm:
            x = self.norm1(input)
            x = x + self.drop1(self.mha(x))
            x = self.norm2(x)
            x = x + self.drop2(self.mlp(x))
        elif order is LayerOrder.PostNoem:
            # multi-head attention
            x = x + self.drop1(self.mha(input))
            x = self.norm1(x)
            # feed-forward layer
            x = x + self.drop2(self.mlp(x))
            x = self.norm2(x)
        else:
            raise ValueError("Unknown order: {:}".format(order))
        return x
