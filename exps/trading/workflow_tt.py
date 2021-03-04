#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
# Refer to:
# - https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.ipynb
# - https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.py
# python exps/trading/workflow_tt.py
#####################################################
import sys, site, argparse
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import qlib
from qlib.config import C
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict


def main(xargs):
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
                    "instruments": xargs.market,
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                    "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
                },
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    }

    model_config = {
        "class": "QuantTransformer",
        "module_path": "trade_models",
        "kwargs": {
            "loss": "mse",
            "GPU": "0",
            "metric": "loss",
        },
    }

    task = {"model": model_config, "dataset": dataset_config}

    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    # start exp to train model
    with R.start(experiment_name="train_tt_model"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)

        # prediction
        recorder = R.get_recorder()
        print(recorder)
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config)
        par.generate()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vanilla Transformable Transformer")
    parser.add_argument("--save_dir", type=str, default="./outputs/tt-ml-runs", help="The checkpoint directory.")
    parser.add_argument("--market", type=str, default="csi300", help="The market indicator.")
    args = parser.parse_args()

    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    exp_manager = C.exp_manager
    exp_manager["kwargs"]["uri"] = "file:{:}".format(Path(args.save_dir).resolve())
    qlib.init(provider_uri=provider_uri, region=REG_CN, exp_manager=exp_manager)

    main(args)
