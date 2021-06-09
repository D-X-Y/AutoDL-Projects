#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from xautodl import spaces
from .super_module import IntSpaceType
from .super_module import BoolSpaceType
from .super_module import LayerOrder
from .super_module import SuperModule
from .super_linear import SuperMLPv2
from .super_norm import SuperLayerNorm1D
from .super_attention import SuperSelfAttention


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
        d_model: IntSpaceType,
        num_heads: IntSpaceType,
        qkv_bias: BoolSpaceType = False,
        mlp_hidden_multiplier: IntSpaceType = 4,
        dropout: Optional[float] = None,
        att_dropout: Optional[float] = None,
        norm_affine: bool = True,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        order: LayerOrder = LayerOrder.PreNorm,
        use_mask: bool = False,
    ):
        super(SuperTransformerEncoderLayer, self).__init__()
        mha = SuperSelfAttention(
            d_model,
            d_model,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=att_dropout,
            proj_drop=None,
            use_mask=use_mask,
        )
        mlp = SuperMLPv2(
            d_model,
            hidden_multiplier=mlp_hidden_multiplier,
            out_features=d_model,
            act_layer=act_layer,
            drop=dropout,
        )
        if order is LayerOrder.PreNorm:
            self.norm1 = SuperLayerNorm1D(d_model, elementwise_affine=norm_affine)
            self.mha = mha
            self.drop = nn.Dropout(dropout or 0.0)
            self.norm2 = SuperLayerNorm1D(d_model, elementwise_affine=norm_affine)
            self.mlp = mlp
        elif order is LayerOrder.PostNorm:
            self.mha = mha
            self.drop1 = nn.Dropout(dropout or 0.0)
            self.norm1 = SuperLayerNorm1D(d_model, elementwise_affine=norm_affine)
            self.mlp = mlp
            self.drop2 = nn.Dropout(dropout or 0.0)
            self.norm2 = SuperLayerNorm1D(d_model, elementwise_affine=norm_affine)
        else:
            raise ValueError("Unknown order: {:}".format(order))
        self._order = order

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

    def forward_candidate(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(inputs)

    def forward_raw(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._order is LayerOrder.PreNorm:
            # https://github.com/google-research/vision_transformer/blob/master/vit_jax/models.py#L135
            x = self.norm1(inputs)
            x = self.mha(x)
            x = self.drop(x)
            x = x + inputs
            # feed-forward layer -- MLP
            y = self.norm2(x)
            outs = x + self.mlp(y)
        elif self._order is LayerOrder.PostNorm:
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
            # multi-head attention
            x = self.mha(inputs)
            x = inputs + self.drop1(x)
            x = self.norm1(x)
            # feed-forward layer -- MLP
            y = self.mlp(x)
            y = x + self.drop2(y)
            outs = self.norm2(y)
        else:
            raise ValueError("Unknown order: {:}".format(self._order))
        return outs
