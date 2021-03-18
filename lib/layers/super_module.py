#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################

import abc
import torch.nn as nn
from enum import Enum


class SuperRunMode(Enum):
    """This class defines the enumerations for Super Model Running Mode."""

    FullModel = "fullmodel"
    Default = "fullmodel"


class SuperModule(abc.ABCMeta, nn.Module):
    """This class equips the nn.Module class with the ability to apply AutoDL."""

    def __init__(self):
        super(SuperModule, self).__init__()
        self._super_run_type = SuperRunMode.default

    @abc.abstractmethod
    def abstract_search_space(self):
        raise NotImplementedError

    @property
    def super_run_type(self):
        return self._super_run_type

    @abc.abstractmethod
    def forward_raw(self, *inputs):
        raise NotImplementedError

    def forward(self, *inputs):
        if self.super_run_type == SuperRunMode.FullModel:
            return self.forward_raw(*inputs)
        else:
            raise ModeError(
                "Unknown Super Model Run Mode: {:}".format(self.super_run_type)
            )
