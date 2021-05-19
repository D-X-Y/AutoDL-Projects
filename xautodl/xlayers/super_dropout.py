#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable, Tuple

from xautodl import spaces
from .super_module import SuperModule
from .super_module import IntSpaceType
from .super_module import BoolSpaceType


class SuperDropout(SuperModule):
    """Applies a the dropout function element-wise."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(SuperDropout, self).__init__()
        self._p = p
        self._inplace = inplace

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout(input, self._p, self.training, self._inplace)

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)

    def extra_repr(self) -> str:
        xstr = "inplace=True" if self._inplace else ""
        return "p={:}".format(self._p) + ", " + xstr


class SuperDrop(SuperModule):
    """Applies a the drop-path function element-wise."""

    def __init__(self, p: float, dims: Tuple[int], recover: bool = True) -> None:
        super(SuperDrop, self).__init__()
        self._p = p
        self._dims = dims
        self._recover = recover

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self._p <= 0:
            return input
        keep_prob = 1 - self._p
        shape = [input.shape[0]] + [
            x if y == -1 else y for x, y in zip(input.shape[1:], self._dims)
        ]
        random_tensor = keep_prob + torch.rand(
            shape, dtype=input.dtype, device=input.device
        )
        random_tensor.floor_()  # binarize
        if self._recover:
            return input.div(keep_prob) * random_tensor
        else:
            return input * random_tensor  # as masks

    def forward_with_container(self, input, container, prefix=[]):
        return self.forward_raw(input)

    def extra_repr(self) -> str:
        return (
            "p={:}".format(self._p)
            + ", dims={:}".format(self._dims)
            + ", recover={:}".format(self._recover)
        )
