#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################

import abc
import torch.nn as nn
from enum import Enum

import spaces


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

    def set_super_run_type(self, super_run_type):
        def _reset_super_run(m):
            if isinstance(m, SuperModule):
                m._super_run_type = super_run_type

        self.apply(_reset_super_run)

    def apply_candiate(self, abstract_child):
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

    @abc.abstractmethod
    def forward_raw(self, *inputs):
        """Use the largest candidate for forward. Similar to the original PyTorch model."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_candidate(self, *inputs):
        raise NotImplementedError

    def forward(self, *inputs):
        if self.super_run_type == SuperRunMode.FullModel:
            return self.forward_raw(*inputs)
        elif self.super_run_type == SuperRunMode.Candidate:
            return self.forward_candidate(*inputs)
        else:
            raise ModeError(
                "Unknown Super Model Run Mode: {:}".format(self.super_run_type)
            )
