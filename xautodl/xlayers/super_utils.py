#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################

import abc
import warnings
from typing import Optional, Union, Callable
import torch
import torch.nn as nn
from enum import Enum

from xautodl import spaces

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


class ShapeContainer:
    """A class to maintain the shape of each weight tensor for a model."""

    def __init__(self):
        self._names = []
        self._shapes = []
        self._name2index = dict()
        self._param_or_buffers = []

    @property
    def shapes(self):
        return self._shapes

    def __getitem__(self, index):
        return self._shapes[index]

    def translate(self, tensors, all_none_match=True):
        result = TensorContainer()
        for index, name in enumerate(self._names):
            cur_num = tensors[index].numel()
            expected_num = self._shapes[index].numel()
            if cur_num < expected_num or (
                cur_num > expected_num and not all_none_match
            ):
                raise ValueError("Invalid {:} vs {:}".format(cur_num, expected_num))
            cur_tensor = tensors[index].view(-1)[:expected_num]
            new_tensor = torch.reshape(cur_tensor, self._shapes[index])
            result.append(name, new_tensor, self._param_or_buffers[index])
        return result

    def append(self, name, shape, param_or_buffer):
        if not isinstance(shape, torch.Size):
            raise TypeError(
                "The input tensor must be torch.Size instead of {:}".format(type(shape))
            )
        self._names.append(name)
        self._shapes.append(shape)
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
        return self._shapes[index]

    def has(self, name):
        return name in self._name2index

    def has_prefix(self, prefix):
        for name, idx in self._name2index.items():
            if name.startswith(prefix):
                return name
        return False

    def numel(self, index=None):
        if index is None:
            shapes = self._shapes
        else:
            shapes = [self._shapes[index]]
        total = 0
        for shape in shapes:
            total += shape.numel()
        return total

    def __len__(self):
        return len(self._names)

    def __repr__(self):
        return "{name}({num} tensors)".format(
            name=self.__class__.__name__, num=len(self)
        )


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

    def create_container(self, tensors):
        result = TensorContainer()
        for index, name in enumerate(self._names):
            new_tensor = tensors[index]
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

    def to_shape_container(self):
        result = ShapeContainer()
        for index, name in enumerate(self._names):
            result.append(
                name, self._tensors[index].shape, self._param_or_buffers[index]
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
