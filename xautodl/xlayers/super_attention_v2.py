#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import math
from typing import Optional, Text

import torch
import torch.nn as nn
import torch.nn.functional as F


from xautodl import spaces
from .super_module import SuperModule
from .super_module import IntSpaceType
from .super_module import BoolSpaceType
from .super_linear import SuperLinear


class SuperQKVAttentionV2(SuperModule):
    """The super model for attention layer."""

    def __init__(
        self,
        qk_att_dim: int,
        in_v_dim: int,
        hidden_dim: int,
        num_heads: int,
        proj_dim: int,
        qkv_bias: bool = False,
        attn_drop: Optional[float] = None,
        proj_drop: Optional[float] = None,
    ):
        super(SuperQKVAttentionV2, self).__init__()
        self._in_v_dim = in_v_dim
        self._qk_att_dim = qk_att_dim
        self._proj_dim = proj_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._qkv_bias = qkv_bias

        self.qk_fc = SuperLinear(qk_att_dim, num_heads, bias=qkv_bias)
        self.v_fc = SuperLinear(in_v_dim, hidden_dim * num_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop or 0.0)
        self.proj = SuperLinear(hidden_dim * num_heads, proj_dim)
        self.proj_drop = nn.Dropout(proj_drop or 0.0)
        self._infinity = 1e9

    @property
    def num_heads(self):
        return spaces.get_max(self._num_heads)

    @property
    def in_v_dim(self):
        return spaces.get_max(self._in_v_dim)

    @property
    def qk_att_dim(self):
        return spaces.get_max(self._qk_att_dim)

    @property
    def hidden_dim(self):
        return spaces.get_max(self._hidden_dim)

    @property
    def proj_dim(self):
        return spaces.get_max(self._proj_dim)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        raise NotImplementedError

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperQKVAttentionV2, self).apply_candidate(abstract_child)
        raise NotImplementedError

    def forward_qkv(
        self, qk_att_tensor, v_tensor, num_head: int, mask=None
    ) -> torch.Tensor:
        qk_att = self.qk_fc(qk_att_tensor)
        B, N, S, _ = qk_att.shape
        assert _ == num_head
        attn_v1 = qk_att.permute(0, 3, 1, 2)
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=1)
            attn_v1 = attn_v1.masked_fill(mask, -self._infinity)
        attn_v1 = attn_v1.softmax(dim=-1)  # B * #head * N * S
        attn_v1 = self.attn_drop(attn_v1)

        v = self.v_fc(v_tensor)
        B0, _, _ = v.shape
        v_v1 = v.reshape(B0, S, num_head, -1).permute(0, 2, 1, 3)
        feats_v1 = (attn_v1 @ v_v1).permute(0, 2, 1, 3).reshape(B, N, -1)
        return feats_v1

    def forward_candidate(self, qk_att_tensor, v_tensor, mask=None) -> torch.Tensor:
        return self.forward_raw(qk_att_tensor, v_tensor, mask)

    def forward_raw(self, qk_att_tensor, v_tensor, mask=None) -> torch.Tensor:
        feats = self.forward_qkv(qk_att_tensor, v_tensor, self.num_heads, mask)
        outs = self.proj(feats)
        outs = self.proj_drop(outs)
        return outs

    def extra_repr(self) -> str:
        return "input_dim={:}, hidden_dim={:}, proj_dim={:}, num_heads={:}, infinity={:}".format(
            (self.qk_att_dim, self.in_v_dim),
            self._hidden_dim,
            self._proj_dim,
            self._num_heads,
            self._infinity,
        )
