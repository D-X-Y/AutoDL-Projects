#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.06 #
#####################################################
from .module_utils import call_by_dict
from .module_utils import call_by_yaml
from .module_utils import nested_call_by_dict
from .module_utils import nested_call_by_yaml
from .yaml_utils import load_yaml

from .torch_utils import count_parameters

from .logger_utils import Logger

# sampler
from .sampler_utils import BatchSampler

# scheduler related
from .scheduler_utils import CosineParamScheduler, WarmupParamScheduler, LRMultiplier


def get_scheduler(indicator, lr):
    if indicator == "warm-cos":
        multiplier = WarmupParamScheduler(
            CosineParamScheduler(lr, lr * 1e-3),
            warmup_factor=0.001,
            warmup_length=0.05,
            warmup_method="linear",
        )

    else:
        raise ValueError("Unknown indicator: {:}".format(indicator))
    return multiplier
