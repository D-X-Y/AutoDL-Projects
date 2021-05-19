#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################

import inspect
import os
import pprint
import logging
from copy import deepcopy

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.log import get_module_logger


def set_log_basic_config(filename=None, format=None, level=None):
    """
    Set the basic configuration for the logging system.
    See details at https://docs.python.org/3/library/logging.html#logging.basicConfig
    :param filename: str or None
        The path to save the logs.
    :param format: the logging format
    :param level: int
    :return: Logger
        Logger object.
    """
    from qlib.config import C

    if level is None:
        level = C.logging_level

    if format is None:
        format = C.logging_config["formatters"]["logger_format"]["format"]

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=filename, format=format, level=level)


def update_gpu(config, gpu):
    config = deepcopy(config)
    if "task" in config and "model" in config["task"]:
        if "GPU" in config["task"]["model"]:
            config["task"]["model"]["GPU"] = gpu
        elif (
            "kwargs" in config["task"]["model"]
            and "GPU" in config["task"]["model"]["kwargs"]
        ):
            config["task"]["model"]["kwargs"]["GPU"] = gpu
    elif "model" in config:
        if "GPU" in config["model"]:
            config["model"]["GPU"] = gpu
        elif "kwargs" in config["model"] and "GPU" in config["model"]["kwargs"]:
            config["model"]["kwargs"]["GPU"] = gpu
    elif "kwargs" in config and "GPU" in config["kwargs"]:
        config["kwargs"]["GPU"] = gpu
    elif "GPU" in config:
        config["GPU"] = gpu
    return config


def update_market(config, market):
    config = deepcopy(config.copy())
    config["market"] = market
    config["data_handler_config"]["instruments"] = market
    return config


def run_exp(
    task_config,
    dataset,
    experiment_name,
    recorder_name,
    uri,
    model_obj_name="model.pkl",
):

    model = init_instance_by_config(task_config["model"])
    model_fit_kwargs = dict(dataset=dataset)

    # Let's start the experiment.
    with R.start(
        experiment_name=experiment_name,
        recorder_name=recorder_name,
        uri=uri,
        resume=True,
    ):
        # Setup log
        recorder_root_dir = R.get_recorder().get_local_dir()
        log_file = os.path.join(recorder_root_dir, "{:}.log".format(experiment_name))

        set_log_basic_config(log_file)
        logger = get_module_logger("q.run_exp")
        logger.info("task_config::\n{:}".format(pprint.pformat(task_config, indent=2)))
        logger.info("[{:}] - [{:}]: {:}".format(experiment_name, recorder_name, uri))
        logger.info("dataset={:}".format(dataset))

        # Train model
        try:
            if hasattr(model, "to"):  # Recoverable model
                ori_device = model.device
                model = R.load_object(model_obj_name)
                model.to(ori_device)
            else:
                model = R.load_object(model_obj_name)
            logger.info("[Find existing object from {:}]".format(model_obj_name))
        except OSError:
            R.log_params(**flatten_dict(update_gpu(task_config, None)))
            if "save_path" in inspect.getfullargspec(model.fit).args:
                model_fit_kwargs["save_path"] = os.path.join(
                    recorder_root_dir, "model.ckp"
                )
            elif "save_dir" in inspect.getfullargspec(model.fit).args:
                model_fit_kwargs["save_dir"] = os.path.join(
                    recorder_root_dir, "model-ckps"
                )
            model.fit(**model_fit_kwargs)
            # remove model to CPU for saving
            if hasattr(model, "to"):
                old_device = model.device
                model.to("cpu")
                R.save_objects(**{model_obj_name: model})
                model.to(old_device)
            else:
                R.save_objects(**{model_obj_name: model})
        except Exception as e:
            raise ValueError("Something wrong: {:}".format(e))
        # Get the recorder
        recorder = R.get_recorder()

        # Generate records: prediction, backtest, and analysis
        for record in task_config["record"]:
            record = deepcopy(record)
            if record["class"] == "MultiSegRecord":
                record["kwargs"] = dict(model=model, dataset=dataset, recorder=recorder)
                sr = init_instance_by_config(record)
                sr.generate(**record["generate_kwargs"])
            elif record["class"] == "SignalRecord":
                srconf = {"model": model, "dataset": dataset, "recorder": recorder}
                record["kwargs"].update(srconf)
                sr = init_instance_by_config(record)
                sr.generate()
            else:
                rconf = {"recorder": recorder}
                record["kwargs"].update(rconf)
                ar = init_instance_by_config(record)
                ar.generate()
