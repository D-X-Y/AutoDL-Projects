#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import torch
from tqdm import tqdm
from procedures import prepare_seed, prepare_logger
from datasets.synthetic_core import get_synthetic_env


def lfna_setup(args):
    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)

    cache_path = (
        logger.path(None) / ".." / "env-{:}-info.pth".format(args.env_version)
    ).resolve()
    if cache_path.exists():
        env_info = torch.load(cache_path)
    else:
        env_info = dict()
        dynamic_env = get_synthetic_env(version=args.env_version)
        env_info["total"] = len(dynamic_env)
        for idx, (timestamp, (_allx, _ally)) in enumerate(tqdm(dynamic_env)):
            env_info["{:}-timestamp".format(idx)] = timestamp
            env_info["{:}-x".format(idx)] = _allx
            env_info["{:}-y".format(idx)] = _ally
        env_info["dynamic_env"] = dynamic_env
        torch.save(env_info, cache_path)

    model_kwargs = dict(
        input_dim=1,
        output_dim=1,
        hidden_dim=args.hidden_dim,
        act_cls="leaky_relu",
        norm_cls="identity",
    )
    return logger, env_info, model_kwargs


class TimeData:
    def __init__(self, timestamp, xs, ys):
        self._timestamp = timestamp
        self._xs = xs
        self._ys = ys

    @property
    def x(self):
        return self._xs

    @property
    def y(self):
        return self._ys

    @property
    def timestamp(self):
        return self._timestamp

    def __repr__(self):
        return "{name}(timestamp={:}, with {num} samples)".format(
            name=self.__class__.__name__, timestamp=self._timestamp, num=len(self._xs)
        )
