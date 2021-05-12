#######################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04   #
#######################################################
# Use module in xlayers to construct different models #
#######################################################
from typing import List, Text, Dict, Any
import torch

__all__ = ["get_model"]


from xlayers.super_core import SuperSequential
from xlayers.super_core import SuperLinear
from xlayers.super_core import super_name2norm
from xlayers.super_core import super_name2activation


def get_model(config: Dict[Text, Any], **kwargs):
    model_type = config.get("model_type", "simple_mlp")
    if model_type == "simple_mlp":
        act_cls = super_name2activation[kwargs["act_cls"]]
        norm_cls = super_name2norm[kwargs["norm_cls"]]
        mean, std = kwargs.get("mean", None), kwargs.get("std", None)
        if "hidden_dim" in kwargs:
            hidden_dim1 = kwargs.get("hidden_dim")
            hidden_dim2 = kwargs.get("hidden_dim")
        else:
            hidden_dim1 = kwargs.get("hidden_dim1", 200)
            hidden_dim2 = kwargs.get("hidden_dim2", 100)
        model = SuperSequential(
            norm_cls(mean=mean, std=std),
            SuperLinear(kwargs["input_dim"], hidden_dim1),
            act_cls(),
            SuperLinear(hidden_dim1, hidden_dim2),
            act_cls(),
            SuperLinear(hidden_dim2, kwargs["output_dim"]),
        )
    else:
        raise TypeError("Unkonwn model type: {:}".format(model_type))
    return model