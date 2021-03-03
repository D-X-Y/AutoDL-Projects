import os
import sys
import qlib
import pprint
import numpy as np
import pandas as pd

from qlib import config as qconfig
from qlib.utils import init_instance_by_config

qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=qconfig.REG_CN)

dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": "2008-01-01",
                        "end_time": "2020-08-01",
                        "fit_start_time": "2008-01-01",
                        "fit_end_time": "2014-12-31",
                        "instruments": "csi300",
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
  dataset = init_instance_by_config(dataset_config)
  pprint.pprint(dataset_config)
  pprint.pprint(dataset)
  import pdb; pdb.set_trace()
  print('Complete')
