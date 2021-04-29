#######################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04   #
#######################################################
# Use module in xlayers to construct different models #
#######################################################
from typing import List, Text, Dict, Any
import torch

__all__ = ["get_model"]


from xlayers.super_core import SuperSequential
from xlayers.super_core import SuperSimpleNorm
from xlayers.super_core import SuperLeakyReLU
from xlayers.super_core import SuperLinear


def get_model(config: Dict[Text, Any], **kwargs):
    model_type = config.get("model_type", "simple_mlp")
    if model_type == "simple_mlp":
        model = SuperSequential(
            SuperSimpleNorm(kwargs["mean"], kwargs["std"]),
            SuperLinear(kwargs["input_dim"], 200),
            SuperLeakyReLU(),
            SuperLinear(200, 100),
            SuperLeakyReLU(),
            SuperLinear(100, kwargs["output_dim"]),
        )
    else:
        raise TypeError("Unkonwn model type: {:}".format(model_type))
    return model
