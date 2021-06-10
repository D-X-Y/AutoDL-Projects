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
from .super_dropout import SuperDropout, SuperDrop
from .super_linear import SuperLinear


class SuperSelfAttention(SuperModule):
    """The super model for attention layer."""

    def __init__(
        self,
        input_dim: IntSpaceType,
        proj_dim: Optional[IntSpaceType],
        num_heads: IntSpaceType,
        qkv_bias: BoolSpaceType = False,
        attn_drop: Optional[float] = None,
        proj_drop: Optional[float] = None,
        use_mask=False,
    ):
        super(SuperSelfAttention, self).__init__()
        self._input_dim = input_dim
        self._proj_dim = proj_dim
        self._num_heads = num_heads
        self._qkv_bias = qkv_bias
        self._use_mask = use_mask
        self._infinity = 1e9

        mul_head_dim = (
            spaces.get_max(input_dim) // spaces.get_min(num_heads)
        ) * spaces.get_min(num_heads)
        assert mul_head_dim == spaces.get_max(input_dim)
        self.q_fc = SuperLinear(input_dim, input_dim, bias=qkv_bias)
        self.k_fc = SuperLinear(input_dim, input_dim, bias=qkv_bias)
        self.v_fc = SuperLinear(input_dim, input_dim, bias=qkv_bias)

        self.attn_drop = SuperDrop(attn_drop or 0.0, [-1, -1, -1, -1], recover=True)
        if proj_dim is not None:
            self.proj = SuperLinear(input_dim, proj_dim)
            self.proj_drop = SuperDropout(proj_drop or 0.0)
        else:
            self.proj = None

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
        if not spaces.is_determined(self._num_heads):
            root_node.append("_num_heads", self._num_heads.abstract(reuse_last=True))
        if not spaces.is_determined(space_q):
            root_node.append("q_fc", space_q)
        if not spaces.is_determined(space_k):
            root_node.append("k_fc", space_k)
        if not spaces.is_determined(space_v):
            root_node.append("v_fc", space_v)
        if self.proj is not None:
            space_proj = self.proj.abstract_search_space
            if not spaces.is_determined(space_proj):
                root_node.append("proj", space_proj)
        return root_node

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperSelfAttention, self).apply_candidate(abstract_child)
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
        if self._use_mask:
            mask = torch.triu(
                torch.ones((N, N), dtype=torch.bool, device=input.device), 1
            )
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            attn_v1 = attn_v1.masked_fill(mask, -self._infinity)
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
        if self.proj is None:
            return feats
        else:
            outs = self.proj(feats)
            outs = self.proj_drop(outs)
            return outs

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        feats = self.forward_qkv(input, self.num_heads)
        if self.proj is None:
            return feats
        else:
            outs = self.proj(feats)
            outs = self.proj_drop(outs)
            return outs

    def extra_repr(self) -> str:
        return (
            "input_dim={:}, proj_dim={:}, num_heads={:}, mask={:}, infinity={:}".format(
                self._input_dim,
                self._proj_dim,
                self._num_heads,
                self._use_mask,
                self._infinity,
            )
        )


class SuperQKVAttention(SuperModule):
    """The super model for attention layer."""

    def __init__(
        self,
        in_q_dim: IntSpaceType,
        in_k_dim: IntSpaceType,
        in_v_dim: IntSpaceType,
        proj_dim: IntSpaceType,
        num_heads: IntSpaceType,
        qkv_bias: BoolSpaceType = False,
        attn_drop: Optional[float] = None,
        proj_drop: Optional[float] = None,
    ):
        super(SuperQKVAttention, self).__init__()
        self._in_v_dim = in_v_dim
        self._in_q_dim = in_q_dim
        self._in_k_dim = in_k_dim
        self._proj_dim = proj_dim
        self._num_heads = num_heads
        self._qkv_bias = qkv_bias

        self.q_fc = SuperLinear(in_q_dim, proj_dim, bias=qkv_bias)
        self.k_fc = SuperLinear(in_k_dim, proj_dim, bias=qkv_bias)
        self.v_fc = SuperLinear(in_v_dim, proj_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop or 0.0)
        self.proj = SuperLinear(proj_dim, proj_dim)
        self.proj_drop = nn.Dropout(proj_drop or 0.0)
        self._infinity = 1e9

    @property
    def num_heads(self):
        return spaces.get_max(self._num_heads)

    @property
    def in_v_dim(self):
        return spaces.get_max(self._in_v_dim)

    @property
    def in_q_dim(self):
        return spaces.get_max(self._in_q_dim)

    @property
    def in_k_dim(self):
        return spaces.get_max(self._in_k_dim)

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
        super(SuperQKVAttention, self).apply_candidate(abstract_child)
        if "q_fc" in abstract_child:
            self.q_fc.apply_candidate(abstract_child["q_fc"])
        if "k_fc" in abstract_child:
            self.k_fc.apply_candidate(abstract_child["k_fc"])
        if "v_fc" in abstract_child:
            self.v_fc.apply_candidate(abstract_child["v_fc"])
        if "proj" in abstract_child:
            self.proj.apply_candidate(abstract_child["proj"])

    def forward_qkv(
        self, q_tensor, k_tensor, v_tensor, num_head: int, mask=None
    ) -> torch.Tensor:
        q = self.q_fc(q_tensor)
        B, N, C = q.shape

        k = self.k_fc(k_tensor)
        B0, S, _ = k.shape

        v = self.v_fc(v_tensor)
        assert B0 == v.shape[0] and S == v.shape[1]

        head_dim = C // num_head
        if num_head > C:
            raise ValueError("Invalid num_head [{:}] vs C [{:}]".format(num_head, C))
        q_v1 = (
            q[:, :, : num_head * head_dim]
            .reshape(B, N, num_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        k_v1 = (
            k[:, :, : num_head * head_dim]
            .reshape(B0, S, num_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        # compute the attention map
        attn_v1 = (q_v1 @ k_v1.transpose(-2, -1)) * math.sqrt(head_dim)
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=1)
            attn_v1 = attn_v1.masked_fill(mask, -self._infinity)
        attn_v1 = attn_v1.softmax(dim=-1)  # B * #head * N * S
        attn_v1 = self.attn_drop(attn_v1)

        v_v1 = (
            v[:, :, : num_head * head_dim]
            .reshape(B0, S, num_head, head_dim)
            .permute(0, 2, 1, 3)
        )
        feats_v1 = (attn_v1 @ v_v1).permute(0, 2, 1, 3).reshape(B, N, -1)
        # process the first [num_head * head_dim] part
        if C == head_dim * num_head:
            feats = feats_v1
        else:  # The channels can not be divided by num_head, the remainder forms an additional head
            # [might have bugs, did not check yet]
            q_v2 = q[:, :, num_head * head_dim :]
            k_v2 = k[:, :, num_head * head_dim :]
            v_v2 = v[:, :, num_head * head_dim :]
            attn_v2 = (q_v2 @ k_v2.transpose(-2, -1)) * math.sqrt(q_v2.shape[-1])
            attn_v2 = attn_v2.softmax(dim=-1)
            attn_v2 = self.attn_drop(attn_v2)
            feats_v2 = attn_v2 @ v_v2
            feats = torch.cat([feats_v1, feats_v2], dim=-1)
        return feats

    def forward_candidate(
        self, q_tensor, k_tensor, v_tensor, mask=None
    ) -> torch.Tensor:
        # check the num_heads:
        if not spaces.is_determined(self._num_heads):
            num_heads = self.abstract_child["_num_heads"].value
        else:
            num_heads = spaces.get_determined_value(self._num_heads)
        feats = self.forward_qkv(q_tensor, k_tensor, v_tensor, num_heads, mask)
        outs = self.proj(feats)
        outs = self.proj_drop(outs)
        return outs

    def forward_raw(self, q_tensor, k_tensor, v_tensor, mask=None) -> torch.Tensor:
        feats = self.forward_qkv(q_tensor, k_tensor, v_tensor, self.num_heads, mask)
        outs = self.proj(feats)
        outs = self.proj_drop(outs)
        return outs

    def extra_repr(self) -> str:
        return "input_dim={:}, proj_dim={:}, num_heads={:}, infinity={:}".format(
            (self.in_q_dim, self.in_k_dim, self.in_v_dim),
            self._proj_dim,
            self._num_heads,
            self._infinity,
        )
