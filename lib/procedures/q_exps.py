#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.log import set_log_basic_config


def update_gpu(config, gpu):
    config = config.copy()
    if "task" in config and "GPU" in config["task"]["model"]:
        config["task"]["model"]["GPU"] = gpu
    elif "model" in config and "GPU" in config["model"]:
        config["model"]["GPU"] = gpu
    elif "GPU" in config:
        config["GPU"] = gpu
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
