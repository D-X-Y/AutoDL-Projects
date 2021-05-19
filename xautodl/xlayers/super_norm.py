#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

from xautodl import spaces
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

    def forward_with_container(self, input, container, prefix=[]):
        super_weight_name = ".".join(prefix + ["weight"])
        if container.has(super_weight_name):
            weight = container.query(super_weight_name)
        else:
            weight = None
        super_bias_name = ".".join(prefix + ["bias"])
        if container.has(super_bias_name):
            bias = container.query(super_bias_name)
        else:
            bias = None
        return F.layer_norm(input, (self.in_dim,), weight, bias, self.eps)

    def extra_repr(self) -> str:
        return (
            "shape={in_dim}, eps={eps}, elementwise_affine={elementwise_affine}".format(
                in_dim=self._in_dim,
                eps=self._eps,
                elementwise_affine=self._elementwise_affine,
            )
        )


class SuperSimpleNorm(SuperModule):
    """Super simple normalization."""

    def __init__(self, mean, std, inplace=False) -> None:
        super(SuperSimpleNorm, self).__init__()
        self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("_std", torch.tensor(std, dtype=torch.float))
        self._inplace = inplace

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        # check inputs ->
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        if not self._inplace:
            tensor = input.clone()
        else:
            tensor = input
        mean = torch.as_tensor(self._mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self._std, dtype=tensor.dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(
                "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                    tensor.dtype
                )
            )
        while mean.ndim < tensor.ndim:
            mean, std = torch.unsqueeze(mean, dim=0), torch.unsqueeze(std, dim=0)
        return tensor.sub_(mean).div_(std)

    def extra_repr(self) -> str:
        return "mean={mean}, std={std}, inplace={inplace}".format(
            mean=self._mean.item(), std=self._std.item(), inplace=self._inplace
        )


class SuperSimpleLearnableNorm(SuperModule):
    """Super simple normalization."""

    def __init__(self, mean=0, std=1, eps=1e-6, inplace=False) -> None:
        super(SuperSimpleLearnableNorm, self).__init__()
        self.register_parameter(
            "_mean", nn.Parameter(torch.tensor(mean, dtype=torch.float))
        )
        self.register_parameter(
            "_std", nn.Parameter(torch.tensor(std, dtype=torch.float))
        )
        self._eps = eps
        self._inplace = inplace

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        # check inputs ->
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        if not self._inplace:
            tensor = input.clone()
        else:
            tensor = input
        mean, std = (
            self._mean.to(tensor.device),
            torch.abs(self._std.to(tensor.device)) + self._eps,
        )
        if (std == 0).any():
            raise ValueError("std leads to division by zero.")
        while mean.ndim < tensor.ndim:
            mean, std = torch.unsqueeze(mean, dim=0), torch.unsqueeze(std, dim=0)
        return tensor.sub_(mean).div_(std)

    def forward_with_container(self, input, container, prefix=[]):
        if not self._inplace:
            tensor = input.clone()
        else:
            tensor = input
        mean_name = ".".join(prefix + ["_mean"])
        std_name = ".".join(prefix + ["_std"])
        mean, std = (
            container.query(mean_name).to(tensor.device),
            torch.abs(container.query(std_name).to(tensor.device)) + self._eps,
        )
        while mean.ndim < tensor.ndim:
            mean, std = torch.unsqueeze(mean, dim=0), torch.unsqueeze(std, dim=0)
        return tensor.sub_(mean).div_(std)

    def extra_repr(self) -> str:
        return "mean={mean}, std={std}, inplace={inplace}".format(
            mean=self._mean.item(), std=self._std.item(), inplace=self._inplace
        )


class SuperIdentity(SuperModule):
    """Super identity mapping layer."""

    def __init__(self, inplace=False, **kwargs) -> None:
        super(SuperIdentity, self).__init__()
        self._inplace = inplace

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        # check inputs ->
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        if not self._inplace:
            tensor = input.clone()
        else:
            tensor = input
        return tensor

    def extra_repr(self) -> str:
        return "inplace={inplace}".format(inplace=self._inplace)

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)
