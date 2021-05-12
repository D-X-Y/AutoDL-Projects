#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import copy
import torch

from xlayers import super_core
from xlayers import trunc_normal_
from models.xcore import get_model


class HyperNet(super_core.SuperModule):
    def __init__(self, shape_container, input_embeding, return_container=True):
        super(HyperNet, self).__init__()
        self._shape_container = shape_container
        self._num_layers = len(shape_container)
        self._numel_per_layer = []
        for ilayer in range(self._num_layers):
            self._numel_per_layer.append(shape_container[ilayer].numel())

        self.register_parameter(
            "_super_layer_embed",
            torch.nn.Parameter(torch.Tensor(self._num_layers, input_embeding)),
        )
        trunc_normal_(self._super_layer_embed, std=0.02)

        model_kwargs = dict(
            input_dim=input_embeding,
            output_dim=max(self._numel_per_layer),
            hidden_dim=input_embeding * 4,
            act_cls="sigmoid",
            norm_cls="identity",
        )
        self._generator = get_model(dict(model_type="simple_mlp"), **model_kwargs)
        self._return_container = return_container
        print("generator: {:}".format(self._generator))

    def forward_raw(self, input):
        weights = self._generator(self._super_layer_embed)
        if self._return_container:
            weights = torch.split(weights, 1)
            return self._shape_container.translate(weights)
        else:
            return weights

    def forward_candidate(self, input):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return "(_super_layer_embed): {:}".format(list(self._super_layer_embed.shape))
