#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
# Refer to:
# - https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.ipynb
# - https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.py
# python exps/trading/workflow_tt.py --gpu 1 --market csi300
#####################################################
import yaml
import argparse

from xautodl.procedures.q_exps import update_gpu
from xautodl.procedures.q_exps import update_market
from xautodl.procedures.q_exps import run_exp

import qlib
from qlib.config import C
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R


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
                        {
                            "class": "RobustZScoreNorm",
                            "kwargs": {"fields_group": "feature", "clip_outlier": True},
                        },
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
        "module_path": "xautodl.trade_models.quant_transformer",
        "kwargs": {
            "net_config": None,
            "opt_config": None,
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
        {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": dict(),
        },
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

    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.utils import init_instance_by_config

    xconfig = """
model:
        class: SFM
        module_path: qlib.contrib.model.pytorch_sfm
        kwargs:
            d_feat: 6
            hidden_size: 64
            output_dim: 32
            freq_dim: 25
            dropout_W: 0.5
            dropout_U: 0.5
            n_epochs: 20
            lr: 1e-3
            batch_size: 1600
            early_stop: 20
            eval_steps: 5
            loss: mse
            optimizer: adam
            GPU: 0
"""
    xconfig = """
model:
        class: TabnetModel
        module_path: qlib.contrib.model.pytorch_tabnet
        kwargs:
            d_feat: 360
            pretrain: True
"""
    xconfig = """
model:
        class: GRU
        module_path: qlib.contrib.model.pytorch_gru
        kwargs:
            d_feat: 6
            hidden_size: 64
            num_layers: 4
            dropout: 0.0
            n_epochs: 200
            lr: 0.001
            early_stop: 20
            batch_size: 800
            metric: loss
            loss: mse
            GPU: 0
"""
    xconfig = yaml.safe_load(xconfig)
    model = init_instance_by_config(xconfig["model"])
    from xautodl.utils.flop_benchmark import count_parameters_in_MB

    # print(count_parameters_in_MB(model.tabnet_model))
    import pdb

    pdb.set_trace()

    save_dir = "{:}-{:}".format(xargs.save_dir, xargs.market)
    dataset = init_instance_by_config(dataset_config)
    for irun in range(xargs.times):
        xmodel_config = model_config.copy()
        xmodel_config = update_gpu(xmodel_config, xargs.gpu)
        task_config = dict(
            model=xmodel_config, dataset=dataset_config, record=record_config
        )

        run_exp(
            task_config,
            dataset,
            xargs.name,
            "recorder-{:02d}-{:02d}".format(irun, xargs.times),
            save_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vanilla Transformable Transformer")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/vtt-runs",
        help="The checkpoint directory.",
    )
    parser.add_argument(
        "--name", type=str, default="Transformer", help="The experiment name."
    )
    parser.add_argument("--times", type=int, default=10, help="The repeated run times.")
    parser.add_argument(
        "--gpu", type=int, default=0, help="The GPU ID used for train / test."
    )
    parser.add_argument(
        "--market", type=str, default="all", help="The market indicator."
    )
    args = parser.parse_args()

    main(args)
