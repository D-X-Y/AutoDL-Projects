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
from .misc_utils import ParsedExpression
from .super_module import SuperModule
from .super_module import IntSpaceType
from .super_module import BoolSpaceType


class SuperReArrange(SuperModule):
    """Applies the rearrange operation."""

    def __init__(self, pattern, **axes_lengths):
        super(SuperReArrange, self).__init__()

        self._pattern = pattern
        self._axes_lengths = axes_lengths
        axes_lengths = tuple(sorted(self._axes_lengths.items()))
        # Perform initial parsing of pattern and provided supplementary info
        # axes_lengths is a tuple of tuples (axis_name, axis_length)
        left, right = pattern.split("->")
        left = ParsedExpression(left)
        right = ParsedExpression(right)

        import pdb

        pdb.set_trace()
        print("-")

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        return root_node

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        import pdb

        pdb.set_trace()
        raise NotImplementedError

    def extra_repr(self) -> str:
        params = repr(self._pattern)
        for axis, length in self._axes_lengths.items():
            params += ", {}={}".format(axis, length)
        return "{:}".format(params)
