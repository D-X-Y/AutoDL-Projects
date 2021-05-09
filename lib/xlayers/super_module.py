#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################

import abc
import warnings
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


class TensorContainer:
    """A class to maintain both parameters and buffers for a model."""

    def __init__(self):
        self._names = []
        self._tensors = []
        self._param_or_buffers = []
        self._name2index = dict()

    def additive(self, tensors):
        result = TensorContainer()
        for index, name in enumerate(self._names):
            new_tensor = self._tensors[index] + tensors[index]
            result.append(name, new_tensor, self._param_or_buffers[index])
        return result

    def no_grad_clone(self):
        result = TensorContainer()
        with torch.no_grad():
            for index, name in enumerate(self._names):
                result.append(
                    name, self._tensors[index].clone(), self._param_or_buffers[index]
                )
        return result

    def requires_grad_(self, requires_grad=True):
        for tensor in self._tensors:
            tensor.requires_grad_(requires_grad)

    def parameters(self):
        return self._tensors

    @property
    def tensors(self):
        return self._tensors

    def flatten(self, tensors=None):
        if tensors is None:
            tensors = self._tensors
        tensors = [tensor.view(-1) for tensor in tensors]
        return torch.cat(tensors)

    def unflatten(self, tensor):
        tensors, s = [], 0
        for raw_tensor in self._tensors:
            length = raw_tensor.numel()
            x = torch.reshape(tensor[s : s + length], shape=raw_tensor.shape)
            tensors.append(x)
            s += length
        return tensors

    def append(self, name, tensor, param_or_buffer):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "The input tensor must be torch.Tensor instead of {:}".format(
                    type(tensor)
                )
            )
        self._names.append(name)
        self._tensors.append(tensor)
        self._param_or_buffers.append(param_or_buffer)
        assert name not in self._name2index, "The [{:}] has already been added.".format(
            name
        )
        self._name2index[name] = len(self._names) - 1

    def query(self, name):
        if not self.has(name):
            raise ValueError(
                "The {:} is not in {:}".format(name, list(self._name2index.keys()))
            )
        index = self._name2index[name]
        return self._tensors[index]

    def has(self, name):
        return name in self._name2index

    def has_prefix(self, prefix):
        for name, idx in self._name2index.items():
            if name.startswith(prefix):
                return name
        return False

    def numel(self):
        total = 0
        for tensor in self._tensors:
            total += tensor.numel()
        return total

    def __len__(self):
        return len(self._names)

    def __repr__(self):
        return "{name}({num} tensors)".format(
            name=self.__class__.__name__, num=len(self)
        )


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

    def add_module(self, name: str, module: Optional[torch.nn.Module]) -> None:
        if not isinstance(module, SuperModule):
            warnings.warn(
                "Add {:}:{:} module, which is not SuperModule, into {:}".format(
                    name, module.__class__.__name__, self.__class__.__name__
                )
                + "\n"
                + "It may cause some functions invalid."
            )
        super(SuperModule, self).add_module(name, module)

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

    def get_w_container(self):
        container = TensorContainer()
        for name, param in self.named_parameters():
            container.append(name, param, True)
        for name, buf in self.named_buffers():
            container.append(name, buf, False)
        return container

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

    def forward_with_container(self, inputs, container, prefix=[]):
        raise NotImplementedError
