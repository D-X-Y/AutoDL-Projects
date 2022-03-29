from dks.base.activation_getter import (
    get_activation_function as _get_numpy_activation_function,
)
from dks.base.activation_transform import _get_activations_params


def subnet_max_func(x, r_fn):
    depth = 7
    res_x = r_fn(x)
    x = r_fn(x)
    for _ in range(depth):
        x = r_fn(r_fn(x)) + x
    return max(x, res_x)


def subnet_max_func_v2(x, r_fn):
    depth = 2
    res_x = r_fn(x)

    x = r_fn(x)
    for _ in range(depth):
        x = 0.8 * r_fn(r_fn(x)) + 0.2 * x

    return max(x, res_x)


def get_transformed_activations(
    activation_names,
    method="TAT",
    dks_params=None,
    tat_params=None,
    max_slope_func=None,
    max_curv_func=None,
    subnet_max_func=None,
    activation_getter=_get_numpy_activation_function,
):
    params = _get_activations_params(
        activation_names,
        method=method,
        dks_params=dks_params,
        tat_params=tat_params,
        max_slope_func=max_slope_func,
        max_curv_func=max_curv_func,
        subnet_max_func=subnet_max_func,
    )
    return params


params = get_transformed_activations(
    ["swish"], method="TAT", subnet_max_func=subnet_max_func
)
print(params)

params = get_transformed_activations(
    ["leaky_relu"], method="TAT", subnet_max_func=subnet_max_func_v2
)
print(params)
