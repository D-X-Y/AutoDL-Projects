import math
from .synthetic_utils import TimeStamp
from .synthetic_env import SyntheticDEnv
from .math_core import LinearFunc
from .math_core import DynamicLinearFunc
from .math_core import DynamicQuadraticFunc
from .math_core import (
    ConstantFunc,
    ComposedSinFunc as SinFunc,
    ComposedCosFunc as CosFunc,
)
from .math_core import GaussianDGenerator


__all__ = ["TimeStamp", "SyntheticDEnv", "get_synthetic_env"]


def get_synthetic_env(total_timestamp=1600, num_per_task=1000, mode=None, version="v1"):
    max_time = math.pi * 10
    if version == "v1":
        mean_generator = ConstantFunc(0)
        std_generator = ConstantFunc(1)
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-2, 2)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=max_time, num=total_timestamp, mode=mode
        )
        oracle_map = DynamicLinearFunc(
            params={
                0: SinFunc(params={0: 2.0, 1: 1.0, 2: 2.2}),  # 2 sin(t) + 2.2
                1: SinFunc(params={0: 1.5, 1: 0.6, 2: 1.8}),  # 1.5 sin(0.6t) + 1.8
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task
        )
    elif version == "v2":
        mean_generator = ConstantFunc(0)
        std_generator = ConstantFunc(1)
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-2, 2)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=max_time, num=total_timestamp, mode=mode
        )
        oracle_map = DynamicQuadraticFunc(
            params={
                0: LinearFunc(params={0: 0.1, 1: 0}),  # 0.1 * t
                1: SinFunc(params={0: 1, 1: 1, 2: 0}),  # sin(t)
                2: ConstantFunc(0),
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task
        )
    elif version.lower() == "v3":
        mean_generator = SinFunc(params={0: 1, 1: 1, 2: 0})  # sin(t)
        std_generator = CosFunc(params={0: 0.5, 1: 1, 2: 1})  # 0.5 cos(t) + 1
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-2, 2)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=max_time, num=total_timestamp, mode=mode
        )
        oracle_map = DynamicQuadraticFunc(
            params={
                0: LinearFunc(params={0: 0.1, 1: 0}),  # 0.1 * t
                1: SinFunc(params={0: 1, 1: 1, 2: 0}),  # sin(t)
                2: ConstantFunc(0),
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task
        )
    else:
        raise ValueError("Unknown version: {:}".format(version))
    return dynamic_env
