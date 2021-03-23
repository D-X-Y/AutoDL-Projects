import os
import sys
import qlib
import pprint
import numpy as np
import pandas as pd

from pathlib import Path
import torch

__file__ = os.path.dirname(os.path.realpath("__file__"))

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
assert lib_dir.exists(), "{:} does not exist".format(lib_dir)
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from trade_models import get_transformer

from qlib import config as qconfig
from qlib.utils import init_instance_by_config
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=qconfig.REG_CN)

dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha360",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": "2008-01-01",
                        "end_time": "2020-08-01",
                        "fit_start_time": "2008-01-01",
                        "fit_end_time": "2014-12-31",
                        "instruments": "csi100",
                    },
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        }
pprint.pprint(dataset_config)
dataset = init_instance_by_config(dataset_config)

df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
model = get_transformer(None)
print(model)

features = torch.from_numpy(df_train["feature"].values).float()
labels = torch.from_numpy(df_train["label"].values).squeeze().float()

batch = list(range(2000))
predicts = model(features[batch])
mask = ~torch.isnan(labels[batch])

pred = predicts[mask]
label = labels[batch][mask]

loss = torch.nn.functional.mse_loss(pred, label)

from sklearn.metrics import mean_squared_error
mse_loss = mean_squared_error(pred.numpy(), label.numpy())
