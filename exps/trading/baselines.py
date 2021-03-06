#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
# python exps/trading/baselines.py --alg GRU
# python exps/trading/baselines.py --alg LSTM
# python exps/trading/baselines.py --alg ALSTM
# python exps/trading/baselines.py --alg XGBoost
# python exps/trading/baselines.py --alg LightGBM
#####################################################
import sys, argparse
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
import ruamel.yaml as yaml

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.log import set_log_basic_config


def retrieve_configs():
    # https://github.com/microsoft/qlib/blob/main/examples/benchmarks/
    config_dir = (lib_dir / ".." / "configs" / "qlib").resolve()
    # algorithm to file names
    alg2names = OrderedDict()
    alg2names["GRU"] = "workflow_config_gru_Alpha360.yaml"
    alg2names["LSTM"] = "workflow_config_lstm_Alpha360.yaml"
    # A dual-stage attention-based recurrent neural network for time series prediction, IJCAI-2017
    alg2names["ALSTM"] = "workflow_config_alstm_Alpha360.yaml"
    # XGBoost: A Scalable Tree Boosting System, KDD-2016
    alg2names["XGBoost"] = "workflow_config_xgboost_Alpha360.yaml"
    # LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NeurIPS-2017
    alg2names["LightGBM"] = "workflow_config_lightgbm_Alpha360.yaml"

    # find the yaml paths
    alg2paths = OrderedDict()
    for idx, (alg, name) in enumerate(alg2names.items()):
        path = config_dir / name
        assert path.exists(), "{:} does not exist.".format(path)
        alg2paths[alg] = str(path)
        print("The {:02d}/{:02d}-th baseline algorithm is {:9s} ({:}).".format(idx, len(alg2names), alg, path))
    return alg2paths


def update_gpu(config, gpu):
    config = config.copy()
    if "GPU" in config["task"]["model"]:
        config["task"]["model"]["GPU"] = gpu
    return config


def update_market(config, market):
    config = config.copy()
    config["market"] = market
    config["data_handler_config"]["instruments"] = market
    return config


def run_exp(task_config, dataset, experiment_name, recorder_name, uri):

    # model initiaiton
    print("")
    print("[{:}] - [{:}]: {:}".format(experiment_name, recorder_name, uri))
    print("dataset={:}".format(dataset))

    model = init_instance_by_config(task_config["model"])

    # start exp
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name, uri=uri):

        log_file = R.get_recorder().root_uri / "{:}.log".format(experiment_name)
        set_log_basic_config(log_file)

        # train model
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        recorder = R.get_recorder()
        R.save_objects(**{"model.pkl": model})

        # generate records: prediction, backtest, and analysis
        for record in task_config["record"]:
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


def main(xargs, exp_yaml):
    assert Path(exp_yaml).exists(), "{:} does not exist.".format(exp_yaml)

    with open(exp_yaml) as fp:
        config = yaml.safe_load(fp)
    config = update_gpu(config, xargs.gpu)
    # config = update_market(config, 'csi300')

    qlib.init(**config.get("qlib_init"))
    dataset_config = config.get("task").get("dataset")
    dataset = init_instance_by_config(dataset_config)
    pprint("args: {:}".format(xargs))
    pprint(dataset_config)
    pprint(dataset)

    for irun in range(xargs.times):
        run_exp(
            config.get("task"), dataset, xargs.alg, "recorder-{:02d}-{:02d}".format(irun, xargs.times), xargs.save_dir
        )


if __name__ == "__main__":

    alg2paths = retrieve_configs()

    parser = argparse.ArgumentParser("Baselines")
    parser.add_argument("--save_dir", type=str, default="./outputs/qlib-baselines", help="The checkpoint directory.")
    parser.add_argument("--times", type=int, default=10, help="The repeated run times.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU ID used for train / test.")
    parser.add_argument("--alg", type=str, choices=list(alg2paths.keys()), required=True, help="The algorithm name.")
    args = parser.parse_args()

    main(args, alg2paths[args.alg])
