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


class SuperReLU(SuperModule):
    """Applies a the rectified linear unit function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        super(SuperReLU, self).__init__()
        self._inplace = inplace

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=self._inplace)

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)

    def extra_repr(self) -> str:
        return "inplace=True" if self._inplace else ""


class SuperGELU(SuperModule):
    """Applies a the Gaussian Error Linear Units function element-wise."""

    def __init__(self) -> None:
        super(SuperGELU, self).__init__()

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)


class SuperSigmoid(SuperModule):
    """Applies a the Sigmoid function element-wise."""

    def __init__(self) -> None:
        super(SuperSigmoid, self).__init__()

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(input)

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)


class SuperLeakyReLU(SuperModule):
    """https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#LeakyReLU"""

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super(SuperLeakyReLU, self).__init__()
        self._negative_slope = negative_slope
        self._inplace = inplace

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(input, self._negative_slope, self._inplace)

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self._inplace else ""
        return "negative_slope={}{}".format(self._negative_slope, inplace_str)


class SuperTanh(SuperModule):
    """Applies a the Tanh function element-wise."""

    def __init__(self) -> None:
        super(SuperTanh, self).__init__()

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input)

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)
