#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from typing import Optional, Text, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from xautodl import spaces
from xautodl.xlayers import weight_init
from xautodl.xlayers import super_core


__all__ = ["DefaultSearchSpace", "DEFAULT_NET_CONFIG", "get_transformer"]


def _get_mul_specs(candidates, num):
    results = []
    for i in range(num):
        results.append(spaces.Categorical(*candidates))
    return results


def _get_list_mul(num, multipler):
    results = []
    for i in range(1, num + 1):
        results.append(i * multipler)
    return results


def _assert_types(x, expected_types):
    if not isinstance(x, expected_types):
        raise TypeError(
            "The type [{:}] is expected to be {:}.".format(type(x), expected_types)
        )


DEFAULT_NET_CONFIG = None
_default_max_depth = 6
DefaultSearchSpace = dict(
    d_feat=6,
    embed_dim=32,
    # embed_dim=spaces.Categorical(*_get_list_mul(8, 16)),
    num_heads=[4] * _default_max_depth,
    mlp_hidden_multipliers=[4] * _default_max_depth,
    qkv_bias=True,
    pos_drop=0.0,
    other_drop=0.0,
)


class SuperTransformer(super_core.SuperModule):
    """The super model for transformer."""

    def __init__(
        self,
        d_feat: int = 6,
        embed_dim: List[super_core.IntSpaceType] = DefaultSearchSpace["embed_dim"],
        num_heads: List[super_core.IntSpaceType] = DefaultSearchSpace["num_heads"],
        mlp_hidden_multipliers: List[super_core.IntSpaceType] = DefaultSearchSpace[
            "mlp_hidden_multipliers"
        ],
        qkv_bias: bool = DefaultSearchSpace["qkv_bias"],
        pos_drop: float = DefaultSearchSpace["pos_drop"],
        other_drop: float = DefaultSearchSpace["other_drop"],
        max_seq_len: int = 65,
    ):
        super(SuperTransformer, self).__init__()
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._mlp_hidden_multipliers = mlp_hidden_multipliers

        # the stem part
        self.input_embed = super_core.SuperAlphaEBDv1(d_feat, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = super_core.SuperPositionalEncoder(
            d_model=embed_dim, max_seq_len=max_seq_len, dropout=pos_drop
        )
        # build the transformer encode layers -->> check params
        _assert_types(num_heads, (tuple, list))
        _assert_types(mlp_hidden_multipliers, (tuple, list))
        assert len(num_heads) == len(mlp_hidden_multipliers), "{:} vs {:}".format(
            len(num_heads), len(mlp_hidden_multipliers)
        )
        # build the transformer encode layers -->> backbone
        layers = []
        for num_head, mlp_hidden_multiplier in zip(num_heads, mlp_hidden_multipliers):
            layer = super_core.SuperTransformerEncoderLayer(
                embed_dim,
                num_head,
                qkv_bias,
                mlp_hidden_multiplier,
                other_drop,
            )
            layers.append(layer)
        self.backbone = super_core.SuperSequential(*layers)

        # the regression head
        self.head = super_core.SuperSequential(
            super_core.SuperLayerNorm1D(embed_dim), super_core.SuperLinear(embed_dim, 1)
        )
        weight_init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @property
    def embed_dim(self):
        return spaces.get_max(self._embed_dim)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        if not spaces.is_determined(self._embed_dim):
            root_node.append("_embed_dim", self._embed_dim.abstract(reuse_last=True))
        xdict = dict(
            input_embed=self.input_embed.abstract_search_space,
            pos_embed=self.pos_embed.abstract_search_space,
            backbone=self.backbone.abstract_search_space,
            head=self.head.abstract_search_space,
        )
        for key, space in xdict.items():
            if not spaces.is_determined(space):
                root_node.append(key, space)
        return root_node

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperTransformer, self).apply_candidate(abstract_child)
        xkeys = ("input_embed", "pos_embed", "backbone", "head")
        for key in xkeys:
            if key in abstract_child:
                getattr(self, key).apply_candidate(abstract_child[key])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            weight_init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, super_core.SuperLinear):
            weight_init.trunc_normal_(m._super_weight, std=0.02)
            if m._super_bias is not None:
                nn.init.constant_(m._super_bias, 0)
        elif isinstance(m, super_core.SuperLayerNorm1D):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        batch, flatten_size = input.shape
        feats = self.input_embed(input)  # batch * 60 * 64
        if not spaces.is_determined(self._embed_dim):
            embed_dim = self.abstract_child["_embed_dim"].value
        else:
            embed_dim = spaces.get_determined_value(self._embed_dim)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        cls_tokens = F.interpolate(
            cls_tokens, size=(embed_dim), mode="linear", align_corners=True
        )
        feats_w_ct = torch.cat((cls_tokens, feats), dim=1)
        feats_w_tp = self.pos_embed(feats_w_ct)
        xfeats = self.backbone(feats_w_tp)
        xfeats = xfeats[:, 0, :]  # use the feature for the first token
        predicts = self.head(xfeats).squeeze(-1)
        return predicts

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        batch, flatten_size = input.shape
        feats = self.input_embed(input)  # batch * 60 * 64
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        feats_w_ct = torch.cat((cls_tokens, feats), dim=1)
        feats_w_tp = self.pos_embed(feats_w_ct)
        xfeats = self.backbone(feats_w_tp)
        xfeats = xfeats[:, 0, :]  # use the feature for the first token
        predicts = self.head(xfeats).squeeze(-1)
        return predicts


def get_transformer(config):
    if config is None:
        return SuperTransformer(6)
    if not isinstance(config, dict):
        raise ValueError("Invalid Configuration: {:}".format(config))
    name = config.get("name", "basic")
    if name == "basic":
        model = SuperTransformer(
            d_feat=config.get("d_feat"),
            embed_dim=config.get("embed_dim"),
            num_heads=config.get("num_heads"),
            mlp_hidden_multipliers=config.get("mlp_hidden_multipliers"),
            qkv_bias=config.get("qkv_bias"),
            pos_drop=config.get("pos_drop"),
            other_drop=config.get("other_drop"),
        )
    else:
        raise ValueError("Unknown model name: {:}".format(name))
    return model
