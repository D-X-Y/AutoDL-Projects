#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.05 #
#####################################################
import math
from .synthetic_utils import TimeStamp
from .synthetic_env import SyntheticDEnv
from .math_core import LinearFunc
from .math_core import DynamicLinearFunc
from .math_core import DynamicQuadraticFunc
from .math_core import ConstantFunc, ComposedSinFunc
from .math_core import GaussianDGenerator


__all__ = ["TimeStamp", "SyntheticDEnv", "get_synthetic_env"]


def get_synthetic_env(total_timestamp=1000, num_per_task=1000, mode=None, version="v1"):
    if version == "v0":
        mean_generator = ConstantFunc(0)
        std_generator = ConstantFunc(1)
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-2, 2)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=math.pi * 8, num=total_timestamp, mode=mode
        )
        oracle_map = DynamicLinearFunc(
            params={
                0: ComposedSinFunc(params={0: 2.0, 1: 1.0, 2: 2.2}),
                1: ConstantFunc(0),
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task
        )
    elif version == "v1":
        mean_generator = ConstantFunc(0)
        std_generator = ConstantFunc(1)
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-2, 2)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=math.pi * 8, num=total_timestamp, mode=mode
        )
        oracle_map = DynamicLinearFunc(
            params={
                0: ComposedSinFunc(params={0: 2.0, 1: 1.0, 2: 2.2}),
                1: ComposedSinFunc(params={0: 1.5, 1: 0.6, 2: 1.8}),
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task
        )
    else:
        raise ValueError("Unknown version: {:}".format(version))
    return dynamic_env
