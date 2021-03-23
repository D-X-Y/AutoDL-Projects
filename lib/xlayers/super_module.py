#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################

import abc
from typing import Optional, Union, Callable
import torch
import torch.nn as nn
from enum import Enum

import spaces

IntSpaceType = Union[int, spaces.Integer, spaces.Categorical]
BoolSpaceType = Union[bool, spaces.Categorical]


class LayerOrder(Enum):
    """This class defines the enumerations for order of operation in a residual or normalization-based layer."""

    PreNorm = "pre-norm"
    PostNorm = "post-norm"


class SuperRunMode(Enum):
    """This class defines the enumerations for Super Model Running Mode."""

    FullModel = "fullmodel"
    Candidate = "candidate"
    Default = "fullmodel"


class SuperModule(abc.ABC, nn.Module):
    """This class equips the nn.Module class with the ability to apply AutoDL."""

    def __init__(self):
        super(SuperModule, self).__init__()
        self._super_run_type = SuperRunMode.Default
        self._abstract_child = None
        self._verbose = False

    def set_super_run_type(self, super_run_type):
        def _reset_super_run(m):
            if isinstance(m, SuperModule):
                m._super_run_type = super_run_type

        self.apply(_reset_super_run)

    def apply_verbose(self, verbose):
        def _reset_verbose(m):
            if isinstance(m, SuperModule):
                m._verbose = verbose

        self.apply(_reset_verbose)

    def apply_candidate(self, abstract_child):
        if not isinstance(abstract_child, spaces.VirtualNode):
            raise ValueError(
                "Invalid abstract child program: {:}".format(abstract_child)
            )
        self._abstract_child = abstract_child

    @property
    def abstract_search_space(self):
        raise NotImplementedError

    @property
    def super_run_type(self):
        return self._super_run_type

    @property
    def abstract_child(self):
        return self._abstract_child

    @property
    def verbose(self):
        return self._verbose

    @abc.abstractmethod
    def forward_raw(self, *inputs):
        """Use the largest candidate for forward. Similar to the original PyTorch model."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_candidate(self, *inputs):
        raise NotImplementedError

    @property
    def name_with_id(self):
        return "name={:}, id={:}".format(self.__class__.__name__, id(self))

    def get_shape_str(self, tensors):
        if isinstance(tensors, (list, tuple)):
            shapes = [self.get_shape_str(tensor) for tensor in tensors]
            if len(shapes) == 1:
                return shapes[0]
            else:
                return ", ".join(shapes)
        elif isinstance(tensors, (torch.Tensor, nn.Parameter)):
            return str(tuple(tensors.shape))
        else:
            raise TypeError("Invalid input type: {:}.".format(type(tensors)))

    def forward(self, *inputs):
        if self.verbose:
            print(
                "[{:}] inputs shape: {:}".format(
                    self.name_with_id, self.get_shape_str(inputs)
                )
            )
        if self.super_run_type == SuperRunMode.FullModel:
            outputs = self.forward_raw(*inputs)
        elif self.super_run_type == SuperRunMode.Candidate:
            outputs = self.forward_candidate(*inputs)
        else:
            raise ModeError(
                "Unknown Super Model Run Mode: {:}".format(self.super_run_type)
            )
        if self.verbose:
            print(
                "[{:}] outputs shape: {:}".format(
                    self.name_with_id, self.get_shape_str(outputs)
                )
            )
        return outputs
