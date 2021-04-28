import copy
from .synthetic_env import SyntheticDEnv
from .math_dynamic_funcs import DynamicQuadraticFunc
from .math_adv_funcs import ConstantFunc, ComposedSinFunc


def get_synthetic_env(total_timestamp=1000, num_per_task=1000, mode=None):
    mean_generator = ComposedSinFunc()
    std_generator = ComposedSinFunc(min_amplitude=0.5, max_amplitude=1.5)
    dynamic_env = SyntheticDEnv(
        [mean_generator],
        [[std_generator]],
        num_per_task=num_per_task,
        timestamp_config=dict(
            min_timestamp=-0.5, max_timestamp=1.5, num=total_timestamp, mode=mode
        ),
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
    return dynamic_env
