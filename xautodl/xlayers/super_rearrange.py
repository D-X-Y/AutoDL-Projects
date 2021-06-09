#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#############################################################
# Borrow the idea of https://github.com/arogozhnikov/einops #
#############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import itertools
import functools
from collections import OrderedDict
from typing import Optional, Callable

from xautodl import spaces
from .misc_utils import ParsedExpression, AnonymousAxis
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
        difference = set.symmetric_difference(left.identifiers, right.identifiers)
        if difference:
            raise ValueError(
                "Identifiers only on one side of expression (should be on both): {}".format(
                    difference
                )
            )

        # parsing all dimensions to find out lengths
        axis_name2known_length = OrderedDict()
        for composite_axis in left.composition:
            for axis_name in composite_axis:
                if isinstance(axis_name, AnonymousAxis):
                    axis_name2known_length[axis_name] = axis_name.value
                else:
                    axis_name2known_length[axis_name] = None
        for axis_name in right.identifiers:
            if axis_name not in axis_name2known_length:
                if isinstance(axis_name, AnonymousAxis):
                    axis_name2known_length[axis_name] = axis_name.value
                else:
                    axis_name2known_length[axis_name] = None

        axis_name2position = {
            name: position for position, name in enumerate(axis_name2known_length)
        }
        for elementary_axis, axis_length in axes_lengths:
            if not ParsedExpression.check_axis_name(elementary_axis):
                raise ValueError("Invalid name for an axis", elementary_axis)
            if elementary_axis not in axis_name2known_length:
                raise ValueError(
                    "Axis {} is not used in transform".format(elementary_axis)
                )
            axis_name2known_length[elementary_axis] = axis_length

        input_composite_axes = []
        # some of shapes will be inferred later - all information is prepared for faster inference
        for composite_axis in left.composition:
            known = {
                axis
                for axis in composite_axis
                if axis_name2known_length[axis] is not None
            }
            unknown = {
                axis for axis in composite_axis if axis_name2known_length[axis] is None
            }
            if len(unknown) > 1:
                raise ValueError("Could not infer sizes for {}".format(unknown))
            assert len(unknown) + len(known) == len(composite_axis)
            input_composite_axes.append(
                (
                    [axis_name2position[axis] for axis in known],
                    [axis_name2position[axis] for axis in unknown],
                )
            )

        axis_position_after_reduction = {}
        for axis_name in itertools.chain(*left.composition):
            if axis_name in right.identifiers:
                axis_position_after_reduction[axis_name] = len(
                    axis_position_after_reduction
                )

        result_axes_grouping = []
        for composite_axis in right.composition:
            result_axes_grouping.append(
                [axis_name2position[axis] for axis in composite_axis]
            )

        ordered_axis_right = list(itertools.chain(*right.composition))
        axes_permutation = tuple(
            axis_position_after_reduction[axis]
            for axis in ordered_axis_right
            if axis in left.identifiers
        )
        #
        self.input_composite_axes = input_composite_axes
        self.output_composite_axes = result_axes_grouping
        self.elementary_axes_lengths = list(axis_name2known_length.values())
        self.axes_permutation = axes_permutation

    @functools.lru_cache(maxsize=1024)
    def reconstruct_from_shape(self, shape):
        if len(shape) != len(self.input_composite_axes):
            raise ValueError(
                "Expected {} dimensions, got {}".format(
                    len(self.input_composite_axes), len(shape)
                )
            )
        axes_lengths = list(self.elementary_axes_lengths)
        for input_axis, (known_axes, unknown_axes) in enumerate(
            self.input_composite_axes
        ):
            length = shape[input_axis]
            known_product = 1
            for axis in known_axes:
                known_product *= axes_lengths[axis]
            if len(unknown_axes) == 0:
                if (
                    isinstance(length, int)
                    and isinstance(known_product, int)
                    and length != known_product
                ):
                    raise ValueError(
                        "Shape mismatch, {} != {}".format(length, known_product)
                    )
            else:
                if (
                    isinstance(length, int)
                    and isinstance(known_product, int)
                    and length % known_product != 0
                ):
                    raise ValueError(
                        "Shape mismatch, can't divide axis of length {} in chunks of {}".format(
                            length, known_product
                        )
                    )

                (unknown_axis,) = unknown_axes
                axes_lengths[unknown_axis] = length // known_product
        # at this point all axes_lengths are computed (either have values or variables, but not Nones)
        final_shape = []
        for output_axis, grouping in enumerate(self.output_composite_axes):
            lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
            final_shape.append(int(np.prod(lengths)))
        axes_reordering = self.axes_permutation
        return axes_lengths, axes_reordering, final_shape

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        return root_node

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        init_shape, axes_reordering, final_shape = self.reconstruct_from_shape(
            tuple(input.shape)
        )
        tensor = torch.reshape(input, init_shape)
        tensor = tensor.permute(axes_reordering)
        tensor = torch.reshape(tensor, final_shape)
        return tensor

    def extra_repr(self) -> str:
        params = repr(self._pattern)
        for axis, length in self._axes_lengths.items():
            params += ", {}={}".format(axis, length)
        return "{:}".format(params)
