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


class SuperLinear(SuperModule):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`"""

    def __init__(
        self,
        in_features: IntSpaceType,
        out_features: IntSpaceType,
        bias: BoolSpaceType = True,
    ) -> None:
        super(SuperLinear, self).__init__()

        # the raw input args
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        # weights to be optimized
        self._super_weight = torch.nn.Parameter(
            torch.Tensor(self.out_features, self.in_features)
        )
        if self.bias:
            self._super_bias = torch.nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("_super_bias", None)
        self.reset_parameters()

    @property
    def in_features(self):
        return spaces.get_max(self._in_features)

    @property
    def out_features(self):
        return spaces.get_max(self._out_features)

    @property
    def bias(self):
        return spaces.has_categorical(self._bias, True)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        if not spaces.is_determined(self._in_features):
            root_node.append(
                "_in_features", self._in_features.abstract(reuse_last=True)
            )
        if not spaces.is_determined(self._out_features):
            root_node.append(
                "_out_features", self._out_features.abstract(reuse_last=True)
            )
        if not spaces.is_determined(self._bias):
            root_node.append("_bias", self._bias.abstract(reuse_last=True))
        return root_node

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self._super_weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._super_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self._super_bias, -bound, bound)

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        # check inputs ->
        if not spaces.is_determined(self._in_features):
            expected_input_dim = self.abstract_child["_in_features"].value
        else:
            expected_input_dim = spaces.get_determined_value(self._in_features)
        if input.size(-1) != expected_input_dim:
            raise ValueError(
                "Expect the input dim of {:} instead of {:}".format(
                    expected_input_dim, input.size(-1)
                )
            )
        # create the weight matrix
        if not spaces.is_determined(self._out_features):
            out_dim = self.abstract_child["_out_features"].value
        else:
            out_dim = spaces.get_determined_value(self._out_features)
        candidate_weight = self._super_weight[:out_dim, :expected_input_dim]
        # create the bias matrix
        if not spaces.is_determined(self._bias):
            if self.abstract_child["_bias"].value:
                candidate_bias = self._super_bias[:out_dim]
            else:
                candidate_bias = None
        else:
            if spaces.get_determined_value(self._bias):
                candidate_bias = self._super_bias[:out_dim]
            else:
                candidate_bias = None
        return F.linear(input, candidate_weight, candidate_bias)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self._super_weight, self._super_bias)

    def extra_repr(self) -> str:
        return "in_features={:}, out_features={:}, bias={:}".format(
            self.in_features, self.out_features, self.bias
        )


class SuperMLP(SuperModule):
    """An MLP layer: FC -> Activation -> Drop -> FC -> Drop."""

    def __init__(
        self,
        in_features: IntSpaceType,
        hidden_features: IntSpaceType,
        out_features: IntSpaceType,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: Optional[float] = None,
    ):
        super(SuperMLP, self).__init__()
        self._in_features = in_features
        self._hidden_features = hidden_features
        self._out_features = out_features
        self._drop_rate = drop
        self.fc1 = SuperLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = SuperLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop or 0.0)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        space_fc1 = self.fc1.abstract_search_space
        space_fc2 = self.fc2.abstract_search_space
        if not spaces.is_determined(space_fc1):
            root_node.append("fc1", space_fc1)
        if not spaces.is_determined(space_fc2):
            root_node.append("fc2", space_fc2)
        return root_node

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperMLP, self).apply_candidate(abstract_child)
        if "fc1" in abstract_child:
            self.fc1.apply_candidate(abstract_child["fc1"])
        if "fc2" in abstract_child:
            self.fc2.apply_candidate(abstract_child["fc2"])

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self._unified_forward(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return self._unified_forward(input)

    def _unified_forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def extra_repr(self) -> str:
        return "in_features={:}, hidden_features={:}, out_features={:}, drop={:}, fc1 -> act -> drop -> fc2 -> drop,".format(
            self._in_features,
            self._hidden_features,
            self._out_features,
            self._drop_rate,
        )
