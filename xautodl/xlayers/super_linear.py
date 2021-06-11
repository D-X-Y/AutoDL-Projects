#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

from xautodl import spaces
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
        self.register_parameter(
            "_super_weight",
            torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features)),
        )
        if self.bias:
            self.register_parameter(
                "_super_bias", torch.nn.Parameter(torch.Tensor(self.out_features))
            )
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
            self._in_features, self._out_features, self._bias
        )

    def forward_with_container(self, input, container, prefix=[]):
        super_weight_name = ".".join(prefix + ["_super_weight"])
        super_weight = container.query(super_weight_name)
        super_bias_name = ".".join(prefix + ["_super_bias"])
        if container.has(super_bias_name):
            super_bias = container.query(super_bias_name)
        else:
            super_bias = None
        return F.linear(input, super_weight, super_bias)


class SuperMLPv1(SuperModule):
    """An MLP layer: FC -> Activation -> Drop -> FC -> Drop."""

    def __init__(
        self,
        in_features: IntSpaceType,
        hidden_features: IntSpaceType,
        out_features: IntSpaceType,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: Optional[float] = None,
    ):
        super(SuperMLPv1, self).__init__()
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
        super(SuperMLPv1, self).apply_candidate(abstract_child)
        if "fc1" in abstract_child:
            self.fc1.apply_candidate(abstract_child["fc1"])
        if "fc2" in abstract_child:
            self.fc2.apply_candidate(abstract_child["fc2"])

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        x = self.fc1(input)
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


class SuperMLPv2(SuperModule):
    """An MLP layer: FC -> Activation -> Drop -> FC -> Drop."""

    def __init__(
        self,
        in_features: IntSpaceType,
        hidden_multiplier: IntSpaceType,
        out_features: IntSpaceType,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: Optional[float] = None,
    ):
        super(SuperMLPv2, self).__init__()
        self._in_features = in_features
        self._hidden_multiplier = hidden_multiplier
        self._out_features = out_features
        self._drop_rate = drop

        self._create_linear(
            "fc1", self.in_features, int(self.in_features * self.hidden_multiplier)
        )
        self._create_linear(
            "fc2", int(self.in_features * self.hidden_multiplier), self.out_features
        )
        self.act = act_layer()
        self.drop = nn.Dropout(drop or 0.0)
        self.reset_parameters()

    @property
    def in_features(self):
        return spaces.get_max(self._in_features)

    @property
    def hidden_multiplier(self):
        return spaces.get_max(self._hidden_multiplier)

    @property
    def out_features(self):
        return spaces.get_max(self._out_features)

    def _create_linear(self, name, inC, outC):
        self.register_parameter(
            "{:}_super_weight".format(name), torch.nn.Parameter(torch.Tensor(outC, inC))
        )
        self.register_parameter(
            "{:}_super_bias".format(name), torch.nn.Parameter(torch.Tensor(outC))
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.fc1_super_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2_super_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1_super_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc1_super_bias, -bound, bound)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2_super_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc2_super_bias, -bound, bound)

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        if not spaces.is_determined(self._in_features):
            root_node.append(
                "_in_features", self._in_features.abstract(reuse_last=True)
            )
        if not spaces.is_determined(self._hidden_multiplier):
            root_node.append(
                "_hidden_multiplier", self._hidden_multiplier.abstract(reuse_last=True)
            )
        if not spaces.is_determined(self._out_features):
            root_node.append(
                "_out_features", self._out_features.abstract(reuse_last=True)
            )
        return root_node

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
        # create the weight and bias matrix for fc1
        if not spaces.is_determined(self._hidden_multiplier):
            hmul = self.abstract_child["_hidden_multiplier"].value * expected_input_dim
        else:
            hmul = spaces.get_determined_value(self._hidden_multiplier)
        hidden_dim = int(expected_input_dim * hmul)
        _fc1_weight = self.fc1_super_weight[:hidden_dim, :expected_input_dim]
        _fc1_bias = self.fc1_super_bias[:hidden_dim]
        x = F.linear(input, _fc1_weight, _fc1_bias)
        x = self.act(x)
        x = self.drop(x)
        # create the weight and bias matrix for fc2
        if not spaces.is_determined(self._out_features):
            out_dim = self.abstract_child["_out_features"].value
        else:
            out_dim = spaces.get_determined_value(self._out_features)
        _fc2_weight = self.fc2_super_weight[:out_dim, :hidden_dim]
        _fc2_bias = self.fc2_super_bias[:out_dim]
        x = F.linear(x, _fc2_weight, _fc2_bias)
        x = self.drop(x)
        return x

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        x = F.linear(input, self.fc1_super_weight, self.fc1_super_bias)
        x = self.act(x)
        x = self.drop(x)
        x = F.linear(x, self.fc2_super_weight, self.fc2_super_bias)
        x = self.drop(x)
        return x

    def extra_repr(self) -> str:
        return "in_features={:}, hidden_multiplier={:}, out_features={:}, drop={:}, fc1 -> act -> drop -> fc2 -> drop,".format(
            self._in_features,
            self._hidden_multiplier,
            self._out_features,
            self._drop_rate,
        )
