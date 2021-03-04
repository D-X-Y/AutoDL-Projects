#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
# Refer to:
# - https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.ipynb
# - https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.py
# python exps/trading/workflow_tt.py
#####################################################
import sys, argparse
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import qlib
from qlib.config import C
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.log import set_log_basic_config


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

    port_analysis_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.strategy",
            "kwargs": {
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "verbose": False,
            "limit_threshold": 0.095,
            "account": 100000000,
            "benchmark": "SH000300",
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    record_config = [
        {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp", "kwargs": dict()},
        {
            "class": "SigAnaRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": dict(ana_long_short=False, ann_scaler=252),
        },
        {
            "class": "PortAnaRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": dict(config=port_analysis_config),
        },
    ]

    task = dict(model=model_config, dataset=dataset_config, record=record_config)


    # start exp to train model
    with R.start(experiment_name="train_tt_model"):
        set_log_basic_config(R.get_recorder().root_uri / 'log.log')

        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(dataset_config)

        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)

        # prediction
        recorder = R.get_recorder()
        print(recorder)

        for record in task["record"]:
            record = record.copy()
            if record["class"] == "SignalRecord":
                srconf = {"model": model, "dataset": dataset, "recorder": recorder}
                record["kwargs"].update(srconf)
                sr = init_instance_by_config(record)
                sr.generate()
            else:
                rconf = {"recorder": recorder}
                record["kwargs"].update(rconf)
                ar = init_instance_by_config(record)
                ar.generate()


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
