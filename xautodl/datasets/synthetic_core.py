#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.05 #
#####################################################
from .synthetic_utils import TimeStamp
from .synthetic_env import EnvSampler
from .synthetic_env import SyntheticDEnv
from .math_core import LinearFunc
from .math_core import DynamicLinearFunc
from .math_core import DynamicQuadraticFunc
from .math_core import ConstantFunc, ComposedSinFunc


__all__ = ["TimeStamp", "SyntheticDEnv", "get_synthetic_env"]


def get_synthetic_env(total_timestamp=1000, num_per_task=1000, mode=None, version="v1"):
    if version == "v1":
        mean_generator = ConstantFunc(0)
        std_generator = ConstantFunc(1)
    elif version == "v2":
        mean_generator = ComposedSinFunc()
        std_generator = ComposedSinFunc(min_amplitude=0.5, max_amplitude=1.5)
    else:
        raise ValueError("Unknown version: {:}".format(version))
    dynamic_env = SyntheticDEnv(
        [mean_generator],
        [[std_generator]],
        num_per_task=num_per_task,
        timestamp_config=dict(
            min_timestamp=-0.5, max_timestamp=1.5, num=total_timestamp, mode=mode
        ),
    )
    if version == "v1":
        function = DynamicLinearFunc()
        function_param = dict()
        function_param[0] = ComposedSinFunc(
            amplitude_scale=ConstantFunc(3.0),
            num_sin_phase=9,
            sin_speed_use_power=False,
        )
        function_param[1] = ConstantFunc(constant=0.9)
    elif version == "v2":
        function = DynamicQuadraticFunc()
        function_param = dict()
        function_param[0] = ComposedSinFunc(
            num_sin_phase=4, phase_shift=1.0, max_amplitude=1.0
        )
        function_param[1] = ConstantFunc(constant=0.9)
        function_param[2] = ComposedSinFunc(
            num_sin_phase=5, phase_shift=0.4, max_amplitude=0.9
        )
    else:
        raise ValueError("Unknown version: {:}".format(version))

    function.set(function_param)
    # dynamic_env.set_oracle_map(copy.deepcopy(function))
    dynamic_env.set_oracle_map(function)
    return dynamic_env
