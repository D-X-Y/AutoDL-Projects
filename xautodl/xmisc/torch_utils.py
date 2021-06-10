#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.06 #
#####################################################
import torch
import torch.nn as nn
import numpy as np


def count_parameters(model_or_parameters, unit="mb"):
    if isinstance(model_or_parameters, nn.Module):
        counts = sum(np.prod(v.size()) for v in model_or_parameters.parameters())
    elif isinstance(model_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(model_or_parameters, (list, tuple)):
        counts = sum(count_parameters(x, None) for x in models_or_parameters)
    else:
        counts = sum(np.prod(v.size()) for v in model_or_parameters)
    if unit.lower() == "kb" or unit.lower() == "k":
        counts /= 1e3
    elif unit.lower() == "mb" or unit.lower() == "m":
        counts /= 1e6
    elif unit.lower() == "gb" or unit.lower() == "g":
        counts /= 1e9
    elif unit is not None:
        raise ValueError("Unknow unit: {:}".format(unit))
    return counts
