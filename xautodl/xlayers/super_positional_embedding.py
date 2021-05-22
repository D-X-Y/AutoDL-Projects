#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
import torch
import torch.nn as nn
import math

from xautodl import spaces
from .super_module import SuperModule
from .super_module import IntSpaceType


class SuperDynamicPositionE(SuperModule):
    """Applies a positional encoding to the input positions."""

    def __init__(self, dimension: int, scale: float = 1.0) -> None:
        super(SuperDynamicPositionE, self).__init__()

        self._scale = scale
        self._dimension = dimension
        # weights to be optimized
        self.register_buffer(
            "_div_term",
            torch.exp(
                torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension)
            ),
        )

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        return root_node

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        positions = torch.unsqueeze(input * self._scale, dim=-1)
        divisions = torch.reshape(
            self._div_term, [1] * input.ndim + [self._div_term.numel()]
        )
        values = positions / divisions
        embeds = torch.cat((torch.sin(values), torch.cos(values)), dim=-1)
        return embeds

    def extra_repr(self) -> str:
        return "scale={:}, dim={:}".format(self._scale, self._dimension)


class SuperPositionalEncoder(SuperModule):
    """Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L65
    """

    def __init__(self, d_model: IntSpaceType, max_seq_len: int, dropout: float = 0.1):
        super(SuperPositionalEncoder, self).__init__()
        self._d_model = d_model
        # create constant 'pe' matrix with values dependant on
        # pos and i
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self.create_pos_embed(max_seq_len, self.d_model))

    @property
    def d_model(self):
        return spaces.get_max(self._d_model)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        if not spaces.is_determined(self._d_model):
            root_node.append("_d_model", self._d_model.abstract(reuse_last=True))
        return root_node

    def create_pos_embed(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model):
                div = 10000 ** ((i // 2) * 2 / d_model)
                value = pos / div
                if i % 2 == 0:
                    pe[pos, i] = math.sin(value)
                else:
                    pe[pos, i] = math.cos(value)
        return pe.unsqueeze(0)

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        batch, seq, fdim = input.shape[:3]
        embeddings = self.pe[:, :seq]
        if not spaces.is_determined(self._d_model):
            expected_d_model = self.abstract_child["_d_model"].value
        else:
            expected_d_model = spaces.get_determined_value(self._d_model)
        assert fdim == expected_d_model, "{:} vs {:}".format(fdim, expected_d_model)

        embeddings = torch.nn.functional.interpolate(
            embeddings, size=(expected_d_model), mode="linear", align_corners=True
        )
        outs = self.dropout(input + embeddings)
        return outs

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        batch, seq, fdim = input.shape[:3]
        embeddings = self.pe[:, :seq]
        outs = self.dropout(input + embeddings)
        return outs
