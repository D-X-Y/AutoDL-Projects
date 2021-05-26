#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import copy
import torch
from tqdm import tqdm
from xautodl.procedures import prepare_seed, prepare_logger
from xautodl.datasets.synthetic_core import get_synthetic_env


def lfna_setup(args):
    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)
    model_kwargs = dict(
        config=dict(model_type="norm_mlp"),
        input_dim=1,
        output_dim=1,
        hidden_dims=[args.hidden_dim] * 2,
        act_cls="relu",
        norm_cls="layer_norm_1d",
    )
    return logger, model_kwargs


def train_model(model, dataset, lr, epochs):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    best_loss, best_param = None, None
    for _iepoch in range(epochs):
        preds = model(dataset.x)
        optimizer.zero_grad()
        loss = criterion(preds, dataset.y)
        loss.backward()
        optimizer.step()
        # save best
        if best_loss is None or best_loss > loss.item():
            best_loss = loss.item()
            best_param = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_param)
    return best_loss


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
        return "{name}(timestamp={timestamp}, with {num} samples)".format(
            name=self.__class__.__name__, timestamp=self._timestamp, num=len(self._xs)
        )
