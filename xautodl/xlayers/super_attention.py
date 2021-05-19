#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from typing import Optional, Text

import torch
import torch.nn as nn
import torch.nn.functional as F


from xautodl import spaces
from .super_module import SuperModule
from .super_module import IntSpaceType
from .super_module import BoolSpaceType
from .super_linear import SuperLinear


class SuperAttention(SuperModule):
    """The super model for attention layer."""

    def __init__(
        self,
        input_dim: IntSpaceType,
        proj_dim: IntSpaceType,
        num_heads: IntSpaceType,
        qkv_bias: BoolSpaceType = False,
        attn_drop: Optional[float] = None,
        proj_drop: Optional[float] = None,
    ):
        super(SuperAttention, self).__init__()
        self._input_dim = input_dim
        self._proj_dim = proj_dim
        self._num_heads = num_heads
        self._qkv_bias = qkv_bias

        self.q_fc = SuperLinear(input_dim, input_dim, bias=qkv_bias)
        self.k_fc = SuperLinear(input_dim, input_dim, bias=qkv_bias)
        self.v_fc = SuperLinear(input_dim, input_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop or 0.0)
        self.proj = SuperLinear(input_dim, proj_dim)
        self.proj_drop = nn.Dropout(proj_drop or 0.0)

    @property
    def num_heads(self):
        return spaces.get_max(self._num_heads)

    @property
    def input_dim(self):
        return spaces.get_max(self._input_dim)

    @property
    def proj_dim(self):
        return spaces.get_max(self._proj_dim)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        space_q = self.q_fc.abstract_search_space
        space_k = self.k_fc.abstract_search_space
        space_v = self.v_fc.abstract_search_space
        space_proj = self.proj.abstract_search_space
        if not spaces.is_determined(self._num_heads):
            root_node.append("_num_heads", self._num_heads.abstract(reuse_last=True))
        if not spaces.is_determined(space_q):
            root_node.append("q_fc", space_q)
        if not spaces.is_determined(space_k):
            root_node.append("k_fc", space_k)
        if not spaces.is_determined(space_v):
            root_node.append("v_fc", space_v)
        if not spaces.is_determined(space_proj):
            root_node.append("proj", space_proj)
        return root_node

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperAttention, self).apply_candidate(abstract_child)
        if "q_fc" in abstract_child:
            self.q_fc.apply_candidate(abstract_child["q_fc"])
        if "k_fc" in abstract_child:
            self.k_fc.apply_candidate(abstract_child["k_fc"])
        if "v_fc" in abstract_child:
            self.v_fc.apply_candidate(abstract_child["v_fc"])
        if "proj" in abstract_child:
            self.proj.apply_candidate(abstract_child["proj"])

    def forward_qkv(self, input: torch.Tensor, num_head: int) -> torch.Tensor:
        B, N, C = input.shape
        q = self.q_fc(input)
        k = self.k_fc(input)
        v = self.v_fc(input)
        if num_head > C:
            raise ValueError("Invalid num_head [{:}] vs C [{:}]".format(num_head, C))
        head_dim = C // num_head
        # process the first [num_head * head_dim] part
        q_v1 = (
            q[:, :, : num_head * head_dim]
            .reshape(B, N, num_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        k_v1 = (
            k[:, :, : num_head * head_dim]
            .reshape(B, N, num_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        v_v1 = (
            v[:, :, : num_head * head_dim]
            .reshape(B, N, num_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        attn_v1 = (q_v1 @ k_v1.transpose(-2, -1)) * math.sqrt(head_dim)
        attn_v1 = attn_v1.softmax(dim=-1)  # B * #head * N * N
        attn_v1 = self.attn_drop(attn_v1)
        feats_v1 = (attn_v1 @ v_v1).permute(0, 2, 1, 3).reshape(B, N, -1)
        if C == head_dim * num_head:
            feats = feats_v1
        else:  # The channels can not be divided by num_head, the remainder forms an additional head
            q_v2 = q[:, :, num_head * head_dim :]
            k_v2 = k[:, :, num_head * head_dim :]
            v_v2 = v[:, :, num_head * head_dim :]
            attn_v2 = (q_v2 @ k_v2.transpose(-2, -1)) * math.sqrt(q_v2.shape[-1])
            attn_v2 = attn_v2.softmax(dim=-1)
            attn_v2 = self.attn_drop(attn_v2)
            feats_v2 = attn_v2 @ v_v2
            feats = torch.cat([feats_v1, feats_v2], dim=-1)
        return feats

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        # check the num_heads:
        if not spaces.is_determined(self._num_heads):
            num_heads = self.abstract_child["_num_heads"].value
        else:
            num_heads = spaces.get_determined_value(self._num_heads)
        feats = self.forward_qkv(input, num_heads)
        outs = self.proj(feats)
        outs = self.proj_drop(outs)
        return outs

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        feats = self.forward_qkv(input, self.num_heads)
        outs = self.proj(feats)
        outs = self.proj_drop(outs)
        return outs

    def extra_repr(self) -> str:
        return "input_dim={:}, proj_dim={:}, num_heads={:}".format(
            self._input_dim, self._proj_dim, self._num_heads
        )
