#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################

import abc
import torch.nn as nn


class SuperModule(abc.ABCMeta, nn.Module):
    """This class equips the nn.Module class with the ability to apply AutoDL."""

    def __init__(self):
        super(SuperModule, self).__init__()

    @abc.abstractmethod
    def abstract_search_space(self):
        raise NotImplementedError
