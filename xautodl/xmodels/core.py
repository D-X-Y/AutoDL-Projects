#######################################################
# Use module in xlayers to construct different models #
#######################################################
from typing import List, Text, Dict, Any
import torch

__all__ = ["get_model"]


from xautodl.xlayers.super_core import SuperSequential
from xautodl.xlayers.super_core import SuperLinear
from xautodl.xlayers.super_core import SuperDropout
from xautodl.xlayers.super_core import super_name2norm
from xautodl.xlayers.super_core import super_name2activation


def get_model(config: Dict[Text, Any], **kwargs):
    model_type = config.get("model_type", "simple_mlp").lower()
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
    elif model_type == "norm_mlp":
        act_cls = super_name2activation[kwargs["act_cls"]]
        norm_cls = super_name2norm[kwargs["norm_cls"]]
        sub_layers, last_dim = [], kwargs["input_dim"]
        for i, hidden_dim in enumerate(kwargs["hidden_dims"]):
            sub_layers.append(SuperLinear(last_dim, hidden_dim))
            if hidden_dim > 1:
                sub_layers.append(norm_cls(hidden_dim, elementwise_affine=False))
            sub_layers.append(act_cls())
            last_dim = hidden_dim
        sub_layers.append(SuperLinear(last_dim, kwargs["output_dim"]))
        model = SuperSequential(*sub_layers)
    elif model_type == "dual_norm_mlp":
        act_cls = super_name2activation[kwargs["act_cls"]]
        norm_cls = super_name2norm[kwargs["norm_cls"]]
        sub_layers, last_dim = [], kwargs["input_dim"]
        for i, hidden_dim in enumerate(kwargs["hidden_dims"]):
            if i > 0:
                sub_layers.append(norm_cls(last_dim, elementwise_affine=False))
            sub_layers.append(SuperLinear(last_dim, hidden_dim))
            sub_layers.append(SuperDropout(kwargs["dropout"]))
            sub_layers.append(SuperLinear(hidden_dim, hidden_dim))
            sub_layers.append(act_cls())
            last_dim = hidden_dim
        sub_layers.append(SuperLinear(last_dim, kwargs["output_dim"]))
        model = SuperSequential(*sub_layers)
    elif model_type == "quant_transformer":
        raise NotImplementedError
    else:
        raise TypeError("Unkonwn model type: {:}".format(model_type))
    return model
