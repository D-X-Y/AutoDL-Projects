import math
from .synthetic_utils import TimeStamp
from .synthetic_env import SyntheticDEnv
from .math_core import LinearSFunc
from .math_core import LinearDFunc
from .math_core import QuadraticDFunc, SinQuadraticDFunc, BinaryQuadraticDFunc
from .math_core import (
    ConstantFunc,
    ComposedSinSFunc as SinFunc,
    ComposedCosSFunc as CosFunc,
)
from .math_core import UniformDGenerator, GaussianDGenerator


__all__ = ["TimeStamp", "SyntheticDEnv", "get_synthetic_env"]


def get_synthetic_env(total_timestamp=1600, num_per_task=1000, mode=None, version="v1"):
    max_time = math.pi * 10
    if version.lower() == "v1":
        mean_generator = ConstantFunc(0)
        std_generator = ConstantFunc(1)
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-3, 3)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=max_time, num=total_timestamp, mode=mode
        )
        oracle_map = LinearDFunc(
            params={
                0: SinFunc(params={0: 2.0, 1: 1.0, 2: 2.2}),  # 2 sin(t) + 2.2
                1: SinFunc(params={0: 1.5, 1: 0.6, 2: 1.8}),  # 1.5 sin(0.6t) + 1.8
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task, noise=0.1
        )
        dynamic_env.set_regression()
    elif version.lower() == "v2":
        mean_generator = ConstantFunc(0)
        std_generator = ConstantFunc(1)
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-3, 3)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=max_time, num=total_timestamp, mode=mode
        )
        oracle_map = QuadraticDFunc(
            params={
                0: LinearSFunc(params={0: 0.1, 1: 0}),  # 0.1 * t
                1: ConstantFunc(0),
                2: CosFunc(params={0: 4.0, 1: 10, 2: 0}),  # 4 * cos(10 * t)
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task, noise=0.1
        )
        dynamic_env.set_regression()
    elif version.lower() == "v3":
        mean_generator = SinFunc(params={0: 1, 1: 1, 2: 0})  # sin(t)
        std_generator = CosFunc(params={0: 0.5, 1: 1, 2: 1})  # 0.5 cos(t) + 1
        data_generator = GaussianDGenerator(
            [mean_generator], [[std_generator]], (-3, 3)
        )
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=max_time, num=total_timestamp, mode=mode
        )
        oracle_map = SinQuadraticDFunc(
            params={
                0: CosFunc(params={0: 0.5, 1: 1, 2: 1}),  # 0.5 cos(t) + 1
                1: SinFunc(params={0: 1, 1: 1, 2: 0}),  # sin(t)
                2: ConstantFunc(0),
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task, noise=0.05
        )
        dynamic_env.set_regression()
    elif version.lower() == "v4":
        l_generator = ConstantFunc(-2)
        r_generator = ConstantFunc(2)
        data_generator = UniformDGenerator([l_generator] * 2, [r_generator] * 2)
        time_generator = TimeStamp(
            min_timestamp=0, max_timestamp=max_time, num=total_timestamp, mode=mode
        )
        oracle_map = BinaryQuadraticDFunc(
            params={
                0: SinFunc(params={0: 1, 1: 3, 2: 0}),  # sin(3 * t)
                1: CosFunc(params={0: 1, 1: 6, 2: 0}),  # cos(6 * t)
                2: ConstantFunc(0),
            }
        )
        dynamic_env = SyntheticDEnv(
            data_generator, oracle_map, time_generator, num_per_task, noise=None
        )
        dynamic_env.set_classification(2)
    else:
        raise ValueError("Unknown version: {:}".format(version))
    return dynamic_env
