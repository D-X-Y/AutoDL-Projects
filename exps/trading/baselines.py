#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
# python exps/trading/baselines.py --alg MLP        #
# python exps/trading/baselines.py --alg GRU        #
# python exps/trading/baselines.py --alg LSTM       #
# python exps/trading/baselines.py --alg ALSTM      #
# python exps/trading/baselines.py --alg NAIVE-V1   #
# python exps/trading/baselines.py --alg NAIVE-V2   #
#                                                   #
# python exps/trading/baselines.py --alg SFM        #
# python exps/trading/baselines.py --alg XGBoost    #
# python exps/trading/baselines.py --alg LightGBM   #
# python exps/trading/baselines.py --alg DoubleE    #
# python exps/trading/baselines.py --alg TabNet     #
#                                                   #############################
# python exps/trading/baselines.py --alg Transformer
# python exps/trading/baselines.py --alg TSF
# python exps/trading/baselines.py --alg TSF-2x24-drop0_0 --market csi300
# python exps/trading/baselines.py --alg TSF-6x32-drop0_0 --market csi300
#################################################################################
import sys
import copy
from datetime import datetime
import argparse
from collections import OrderedDict
from pprint import pprint
import ruamel.yaml as yaml

from xautodl.config_utils import arg_str2bool
from xautodl.procedures.q_exps import update_gpu
from xautodl.procedures.q_exps import update_market
from xautodl.procedures.q_exps import run_exp

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.utils import flatten_dict


def to_drop(config, pos_drop, other_drop):
    config = copy.deepcopy(config)
    net = config["task"]["model"]["kwargs"]["net_config"]
    net["pos_drop"] = pos_drop
    net["other_drop"] = other_drop
    return config


def to_layer(config, embed_dim, depth):
    config = copy.deepcopy(config)
    net = config["task"]["model"]["kwargs"]["net_config"]
    net["embed_dim"] = embed_dim
    net["num_heads"] = [4] * depth
    net["mlp_hidden_multipliers"] = [4] * depth
    return config


def extend_transformer_settings(alg2configs, name):
    config = copy.deepcopy(alg2configs[name])
    for i in range(1, 9):
        for j in (6, 12, 24, 32, 48, 64):
            for k1 in (0, 0.05, 0.1, 0.2, 0.3):
                for k2 in (0, 0.1):
                    alg2configs[
                        name + "-{:}x{:}-drop{:}_{:}".format(i, j, k1, k2)
                    ] = to_layer(to_drop(config, k1, k2), j, i)
    return alg2configs


def replace_start_time(config, start_time):
    config = copy.deepcopy(config)
    xtime = datetime.strptime(start_time, "%Y-%m-%d")
    config["data_handler_config"]["start_time"] = xtime.date()
    config["data_handler_config"]["fit_start_time"] = xtime.date()
    config["task"]["dataset"]["kwargs"]["segments"]["train"][0] = xtime.date()
    return config


def extend_train_data(alg2configs, name):
    config = copy.deepcopy(alg2configs[name])
    start_times = (
        "2008-01-01",
        "2008-07-01",
        "2009-01-01",
        "2009-07-01",
        "2010-01-01",
        "2011-01-01",
        "2012-01-01",
        "2013-01-01",
    )
    for start_time in start_times:
        config = replace_start_time(config, start_time)
        alg2configs[name + "s{:}".format(start_time)] = config
    return alg2configs


def refresh_record(alg2configs):
    alg2configs = copy.deepcopy(alg2configs)
    for key, config in alg2configs.items():
        xlist = config["task"]["record"]
        new_list = []
        for x in xlist:
            # remove PortAnaRecord and SignalMseRecord
            if x["class"] != "PortAnaRecord" and x["class"] != "SignalMseRecord":
                new_list.append(x)
        ## add MultiSegRecord
        new_list.append(
            {
                "class": "MultiSegRecord",
                "module_path": "qlib.contrib.workflow",
                "generate_kwargs": {
                    "segments": {"train": "train", "valid": "valid", "test": "test"},
                    "save": True,
                },
            }
        )
        config["task"]["record"] = new_list
    return alg2configs


def retrieve_configs():
    # https://github.com/microsoft/qlib/blob/main/examples/benchmarks/
    config_dir = (lib_dir / ".." / "configs" / "qlib").resolve()
    # algorithm to file names
    alg2names = OrderedDict()
    alg2names["GRU"] = "workflow_config_gru_Alpha360.yaml"
    alg2names["LSTM"] = "workflow_config_lstm_Alpha360.yaml"
    alg2names["MLP"] = "workflow_config_mlp_Alpha360.yaml"
    # A dual-stage attention-based recurrent neural network for time series prediction, IJCAI-2017
    alg2names["ALSTM"] = "workflow_config_alstm_Alpha360.yaml"
    # XGBoost: A Scalable Tree Boosting System, KDD-2016
    alg2names["XGBoost"] = "workflow_config_xgboost_Alpha360.yaml"
    # LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NeurIPS-2017
    alg2names["LightGBM"] = "workflow_config_lightgbm_Alpha360.yaml"
    # State Frequency Memory (SFM): Stock Price Prediction via Discovering Multi-Frequency Trading Patterns, KDD-2017
    alg2names["SFM"] = "workflow_config_sfm_Alpha360.yaml"
    # DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis, https://arxiv.org/pdf/2010.01265.pdf
    alg2names["DoubleE"] = "workflow_config_doubleensemble_Alpha360.yaml"
    alg2names["TabNet"] = "workflow_config_TabNet_Alpha360.yaml"
    alg2names["NAIVE-V1"] = "workflow_config_naive_v1_Alpha360.yaml"
    alg2names["NAIVE-V2"] = "workflow_config_naive_v2_Alpha360.yaml"
    alg2names["Transformer"] = "workflow_config_transformer_Alpha360.yaml"
    alg2names["TSF"] = "workflow_config_transformer_basic_Alpha360.yaml"

    # find the yaml paths
    alg2configs = OrderedDict()
    print("Start retrieving the algorithm configurations")
    for idx, (alg, name) in enumerate(alg2names.items()):
        path = config_dir / name
        assert path.exists(), "{:} does not exist.".format(path)
        with open(path) as fp:
            alg2configs[alg] = yaml.safe_load(fp)
        print(
            "The {:02d}/{:02d}-th baseline algorithm is {:9s} ({:}).".format(
                idx, len(alg2configs), alg, path
            )
        )
    alg2configs = extend_transformer_settings(alg2configs, "TSF")
    alg2configs = refresh_record(alg2configs)
    # extend the algorithms by different train-data
    for name in ("TSF-2x24-drop0_0", "TSF-6x32-drop0_0"):
        alg2configs = extend_train_data(alg2configs, name)
    print(
        "There are {:} algorithms : {:}".format(
            len(alg2configs), list(alg2configs.keys())
        )
    )
    return alg2configs


def main(alg_name, market, config, times, save_dir, gpu):

    pprint("Run {:}".format(alg_name))
    config = update_market(config, market)
    config = update_gpu(config, gpu)

    qlib.init(**config.get("qlib_init"))
    dataset_config = config.get("task").get("dataset")
    dataset = init_instance_by_config(dataset_config)
    pprint(dataset_config)
    pprint(dataset)

    for irun in range(times):
        run_exp(
            config.get("task"),
            dataset,
            alg_name,
            "recorder-{:02d}-{:02d}".format(irun, times),
            "{:}-{:}".format(save_dir, market),
        )


if __name__ == "__main__":

    alg2configs = retrieve_configs()

    parser = argparse.ArgumentParser("Baselines")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/qlib-baselines",
        help="The checkpoint directory.",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="all",
        choices=["csi100", "csi300", "all"],
        help="The market indicator.",
    )
    parser.add_argument("--times", type=int, default=5, help="The repeated run times.")
    parser.add_argument(
        "--shared_dataset",
        type=arg_str2bool,
        default=False,
        help="Whether to share the dataset for all algorithms?",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="The GPU ID used for train / test."
    )
    parser.add_argument(
        "--alg",
        type=str,
        choices=list(alg2configs.keys()),
        nargs="+",
        required=True,
        help="The algorithm name(s).",
    )
    args = parser.parse_args()

    if len(args.alg) == 1:
        main(
            args.alg[0],
            args.market,
            alg2configs[args.alg[0]],
            args.times,
            args.save_dir,
            args.gpu,
        )
    elif len(args.alg) > 1:
        assert args.shared_dataset, "Must allow share dataset"
        pprint(args)
        configs = [
            update_gpu(update_market(alg2configs[name], args.market), args.gpu)
            for name in args.alg
        ]
        qlib.init(**configs[0].get("qlib_init"))
        dataset_config = configs[0].get("task").get("dataset")
        dataset = init_instance_by_config(dataset_config)
        pprint(dataset_config)
        pprint(dataset)
        for alg_name, config in zip(args.alg, configs):
            print("Run {:} over {:}".format(alg_name, args.alg))
            for irun in range(args.times):
                run_exp(
                    config.get("task"),
                    dataset,
                    alg_name,
                    "recorder-{:02d}-{:02d}".format(irun, args.times),
                    "{:}-{:}".format(args.save_dir, args.market),
                )
