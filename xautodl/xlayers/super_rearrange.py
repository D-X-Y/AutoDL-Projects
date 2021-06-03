#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#############################################################
# Borrow the idea of https://github.com/arogozhnikov/einops #
#############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

from xautodl import spaces
from .super_module import SuperModule
from .super_module import IntSpaceType
from .super_module import BoolSpaceType


class SuperRearrange(SuperModule):
    """Applies the rearrange operation."""

    def __init__(self, pattern, **axes_lengths):
        super(SuperRearrange, self).__init__()

        self._pattern = pattern
        self._axes_lengths = axes_lengths
        self.reset_parameters()

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        return root_node

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        params = repr(self._pattern)
        for axis, length in self._axes_lengths.items():
            params += ", {}={}".format(axis, length)
        return "{}({})".format(self.__class__.__name__, params)
