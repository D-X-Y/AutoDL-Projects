import os
import sys
import ruamel.yaml as yaml
import pprint
import numpy as np
import pandas as pd
from pathlib import Path

import qlib
from qlib import config as qconfig
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=qconfig.REG_CN)

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
                "instruments": "csi300",
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ],
               "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
            },
        },
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
        },
    },
}

if __name__ == "__main__":

    qlib_root_dir = (Path(__file__).parent / '..' / '..' / '.latent-data' / 'qlib').resolve()
    demo_yaml_path = qlib_root_dir / 'examples' / 'benchmarks' / 'GRU' / 'workflow_config_gru_Alpha360.yaml'
    print('Demo-workflow-yaml: {:}'.format(demo_yaml_path))
    with open(demo_yaml_path, 'r') as fp:
      config = yaml.safe_load(fp)
    pprint.pprint(config['task']['dataset'])
    
    dataset = init_instance_by_config(dataset_config)
    pprint.pprint(dataset_config)
    pprint.pprint(dataset)

    df_train, df_valid, df_test = dataset.prepare(
        ["train", "valid", "test"],
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )

    x_train, y_train = df_train["feature"], df_train["label"]

    import pdb

    pdb.set_trace()
    print("Complete")

