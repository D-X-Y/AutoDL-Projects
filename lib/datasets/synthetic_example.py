#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################

from .math_adv_funcs import DynamicQuadraticFunc
from .math_adv_funcs import ConstantFunc, ComposedSinFunc
from .synthetic_env import SyntheticDEnv


def create_example_v1(
    timestamp_config=dict(num=100, min_timestamp=0.0, max_timestamp=1.0),
    num_per_task=5000,
):
    mean_generator = ComposedSinFunc()
    std_generator = ComposedSinFunc(min_amplitude=0.5, max_amplitude=0.5)

    dynamic_env = SyntheticDEnv(
        [mean_generator],
        [[std_generator]],
        num_per_task=num_per_task,
        timestamp_config=timestamp_config,
    )

    function = DynamicQuadraticFunc()
    function_param = dict()
    function_param[0] = ComposedSinFunc(
        num_sin_phase=4, phase_shift=1.0, max_amplitude=1.0
    )
    function_param[1] = ConstantFunc(constant=0.9)
    function_param[2] = ComposedSinFunc(
        num_sin_phase=5, phase_shift=0.4, max_amplitude=0.9
    )
    function.set(function_param)
    return dynamic_env, function
