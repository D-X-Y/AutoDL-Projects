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
