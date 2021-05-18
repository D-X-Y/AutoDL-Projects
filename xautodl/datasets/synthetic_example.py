#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import copy

from .math_dynamic_funcs import DynamicLinearFunc, DynamicQuadraticFunc
from .math_adv_funcs import ConstantFunc, ComposedSinFunc
from .synthetic_env import SyntheticDEnv


def create_example(timestamp_config=None, num_per_task=5000, indicator="v1"):
    if indicator == "v1":
        return create_example_v1(timestamp_config, num_per_task)
    elif indicator == "v2":
        return create_example_v2(timestamp_config, num_per_task)
    else:
        raise ValueError("Unkonwn indicator: {:}".format(indicator))


def create_example_v1(
    timestamp_config=None,
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

    dynamic_env.set_oracle_map(copy.deepcopy(function))
    return dynamic_env, function


def create_example_v2(
    timestamp_config=None,
    num_per_task=5000,
):
    mean_generator = ConstantFunc(0)
    std_generator = ConstantFunc(1)

    dynamic_env = SyntheticDEnv(
        [mean_generator],
        [[std_generator]],
        num_per_task=num_per_task,
        timestamp_config=timestamp_config,
    )

    function = DynamicLinearFunc()
    function_param = dict()
    function_param[0] = ComposedSinFunc(
        amplitude_scale=ConstantFunc(1.0), period_phase_shift=ConstantFunc(1.0)
    )
    function_param[1] = ConstantFunc(constant=0.9)
    function.set(function_param)

    dynamic_env.set_oracle_map(copy.deepcopy(function))
    return dynamic_env, function
