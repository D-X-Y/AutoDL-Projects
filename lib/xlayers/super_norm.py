#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

import spaces
from .super_module import SuperModule
from .super_module import IntSpaceType
from .super_module import BoolSpaceType


class SuperLayerNorm1D(SuperModule):
    """Super Layer Norm."""

    def __init__(
        self, dim: IntSpaceType, eps: float = 1e-6, elementwise_affine: bool = True
    ) -> None:
        super(SuperLayerNorm1D, self).__init__()
        self._in_dim = dim
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(self.in_dim)))
            self.register_parameter("bias", nn.Parameter(torch.Tensor(self.in_dim)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def in_dim(self):
        return spaces.get_max(self._in_dim)

    @property
    def eps(self):
        return self._eps

    def reset_parameters(self) -> None:
        if self._elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        if not spaces.is_determined(self._in_dim):
            root_node.append("_in_dim", self._in_dim.abstract(reuse_last=True))
        return root_node

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        # check inputs ->
        if not spaces.is_determined(self._in_dim):
            expected_input_dim = self.abstract_child["_in_dim"].value
        else:
            expected_input_dim = spaces.get_determined_value(self._in_dim)
        if input.size(-1) != expected_input_dim:
            raise ValueError(
                "Expect the input dim of {:} instead of {:}".format(
                    expected_input_dim, input.size(-1)
                )
            )
        if self._elementwise_affine:
            weight = self.weight[:expected_input_dim]
            bias = self.bias[:expected_input_dim]
        else:
            weight, bias = None, None
        return F.layer_norm(input, (expected_input_dim,), weight, bias, self.eps)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, (self.in_dim,), self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (
            "shape={in_dim}, eps={eps}, elementwise_affine={elementwise_affine}".format(
                in_dim=self._in_dim,
                eps=self._eps,
                elementwise_affine=self._elementwise_affine,
            )
        )
