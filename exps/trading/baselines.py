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
#                                                   #
# python exps/trading/baselines.py --alg Transformer#
#####################################################
import sys
import argparse
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
import ruamel.yaml as yaml

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from procedures.q_exps import update_gpu
from procedures.q_exps import update_market
from procedures.q_exps import run_exp

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.utils import flatten_dict


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

    # find the yaml paths
    alg2paths = OrderedDict()
    print("Start retrieving the algorithm configurations")
    for idx, (alg, name) in enumerate(alg2names.items()):
        path = config_dir / name
        assert path.exists(), "{:} does not exist.".format(path)
        alg2paths[alg] = str(path)
        print(
            "The {:02d}/{:02d}-th baseline algorithm is {:9s} ({:}).".format(
                idx, len(alg2names), alg, path
            )
        )
    return alg2paths


def main(xargs, exp_yaml):
    assert Path(exp_yaml).exists(), "{:} does not exist.".format(exp_yaml)

    pprint("Run {:}".format(xargs.alg))
    with open(exp_yaml) as fp:
        config = yaml.safe_load(fp)
    config = update_market(config, xargs.market)
    config = update_gpu(config, xargs.gpu)

    qlib.init(**config.get("qlib_init"))
    dataset_config = config.get("task").get("dataset")
    dataset = init_instance_by_config(dataset_config)
    pprint("args: {:}".format(xargs))
    pprint(dataset_config)
    pprint(dataset)

    for irun in range(xargs.times):
        run_exp(
            config.get("task"),
            dataset,
            xargs.alg,
            "recorder-{:02d}-{:02d}".format(irun, xargs.times),
            "{:}-{:}".format(xargs.save_dir, xargs.market),
        )


if __name__ == "__main__":

    alg2paths = retrieve_configs()

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
    parser.add_argument("--times", type=int, default=10, help="The repeated run times.")
    parser.add_argument(
        "--gpu", type=int, default=0, help="The GPU ID used for train / test."
    )
    parser.add_argument(
        "--alg",
        type=str,
        choices=list(alg2paths.keys()),
        required=True,
        help="The algorithm name.",
    )
    args = parser.parse_args()

    main(args, alg2paths[args.alg])
