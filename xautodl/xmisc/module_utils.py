#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################
from typing import Union, Dict, Text, Any
import importlib

from .yaml_utils import load_yaml

CLS_FUNC_KEY = "class_or_func"
KEYS = (CLS_FUNC_KEY, "module_path", "args", "kwargs")


def has_key_words(xdict):
    if not isinstance(xdict, dict):
        return False
    key_set = set(KEYS)
    cur_set = set(xdict.keys())
    return key_set.intersection(cur_set) == key_set


def get_module_by_module_path(module_path):
    """Load the module from the path."""

    if module_path.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location("", module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return module


def call_by_dict(config: Dict[Text, Any], *args, **kwargs) -> object:
    """
    get initialized instance with config
    Parameters
    ----------
    config : a dictionary, such as:
            {
                'cls_or_func': 'ClassName',
                'args': list,
                'kwargs': dict,
                'model_path': a string indicating the path,
            }
    Returns
    -------
    object:
        An initialized object based on the config info
    """
    module = get_module_by_module_path(config["module_path"])
    cls_or_func = getattr(module, config[CLS_FUNC_KEY])
    args = tuple(list(config["args"]) + list(args))
    kwargs = {**config["kwargs"], **kwargs}
    return cls_or_func(*args, **kwargs)


def call_by_yaml(path, *args, **kwargs) -> object:
    config = load_yaml(path)
    return call_by_config(config, *args, **kwargs)


def nested_call_by_dict(config: Union[Dict[Text, Any], Any], *args, **kwargs) -> object:
    """Similar to `call_by_dict`, but differently, the args may contain another dict needs to be called."""
    if isinstance(config, list):
        return [nested_call_by_dict(x) for x in config]
    elif isinstance(config, tuple):
        return (nested_call_by_dict(x) for x in config)
    elif not isinstance(config, dict):
        return config
    elif not has_key_words(config):
        return {key: nested_call_by_dict(x) for x, key in config.items()}
    else:
        module = get_module_by_module_path(config["module_path"])
        cls_or_func = getattr(module, config[CLS_FUNC_KEY])
        args = tuple(list(config["args"]) + list(args))
        kwargs = {**config["kwargs"], **kwargs}
        # check whether there are nested special dict
        new_args = [nested_call_by_dict(x) for x in args]
        new_kwargs = {}
        for key, x in kwargs.items():
            new_kwargs[key] = nested_call_by_dict(x)
        return cls_or_func(*new_args, **new_kwargs)


def nested_call_by_yaml(path, *args, **kwargs) -> object:
    config = load_yaml(path)
    return nested_call_by_dict(config, *args, **kwargs)
