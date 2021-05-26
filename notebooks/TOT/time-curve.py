import os
import re
import sys
import torch
import qlib
import pprint
from collections import OrderedDict
import numpy as np
import pandas as pd

from pathlib import Path

# __file__ = os.path.dirname(os.path.realpath("__file__"))
note_dir = Path(__file__).parent.resolve()
root_dir = (Path(__file__).parent / ".." / "..").resolve()
lib_dir = (root_dir / "lib").resolve()
print("The root path: {:}".format(root_dir))
print("The library path: {:}".format(lib_dir))
assert lib_dir.exists(), "{:} does not exist".format(lib_dir)
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import qlib
from qlib import config as qconfig
from qlib.workflow import R

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=qconfig.REG_CN)

from utils.qlib_utils import QResult


def filter_finished(recorders):
    returned_recorders = dict()
    not_finished = 0
    for key, recorder in recorders.items():
        if recorder.status == "FINISHED":
            returned_recorders[key] = recorder
        else:
            not_finished += 1
    return returned_recorders, not_finished


def add_to_dict(xdict, timestamp, value):
    date = timestamp.date().strftime("%Y-%m-%d")
    if date in xdict:
        raise ValueError("This date [{:}] is already in the dict".format(date))
    xdict[date] = value


def query_info(save_dir, verbose, name_filter, key_map):
    if isinstance(save_dir, list):
        results = []
        for x in save_dir:
            x = query_info(x, verbose, name_filter, key_map)
            results.extend(x)
        return results
    # Here, the save_dir must be a string
    R.set_uri(str(save_dir))
    experiments = R.list_experiments()

    if verbose:
        print("There are {:} experiments.".format(len(experiments)))
    qresults = []
    for idx, (key, experiment) in enumerate(experiments.items()):
        if experiment.id == "0":
            continue
        if (
            name_filter is not None
            and re.fullmatch(name_filter, experiment.name) is None
        ):
            continue
        recorders = experiment.list_recorders()
        recorders, not_finished = filter_finished(recorders)
        if verbose:
            print(
                "====>>>> {:02d}/{:02d}-th experiment {:9s} has {:02d}/{:02d} finished recorders.".format(
                    idx + 1,
                    len(experiments),
                    experiment.name,
                    len(recorders),
                    len(recorders) + not_finished,
                )
            )
        result = QResult(experiment.name)
        for recorder_id, recorder in recorders.items():
            file_names = ["results-train.pkl", "results-valid.pkl", "results-test.pkl"]
            date2IC = OrderedDict()
            for file_name in file_names:
                xtemp = recorder.load_object(file_name)["all-IC"]
                timestamps, values = xtemp.index.tolist(), xtemp.tolist()
                for timestamp, value in zip(timestamps, values):
                    add_to_dict(date2IC, timestamp, value)
            result.update(recorder.list_metrics(), key_map)
            result.append_path(
                os.path.join(recorder.uri, recorder.experiment_id, recorder.id)
            )
            result.append_date2ICs(date2IC)
        if not len(result):
            print("There are no valid recorders for {:}".format(experiment))
            continue
        else:
            if verbose:
                print(
                    "There are {:} valid recorders for {:}".format(
                        len(recorders), experiment.name
                    )
                )
        qresults.append(result)
    return qresults


##
paths = [root_dir / "outputs" / "qlib-baselines-csi300"]
paths = [path.resolve() for path in paths]
print(paths)

key_map = dict()
for xset in ("train", "valid", "test"):
    key_map["{:}-mean-IC".format(xset)] = "IC ({:})".format(xset)
    key_map["{:}-mean-ICIR".format(xset)] = "ICIR ({:})".format(xset)
qresults = query_info(paths, False, "TSF-2x24-drop0_0s.*-.*-01", key_map)
print("Find {:} results".format(len(qresults)))
times = []
for qresult in qresults:
    times.append(qresult.name.split("0_0s")[-1])
print(times)
save_path = os.path.join(note_dir, "temp-time-x.pth")
torch.save(qresults, save_path)
print(save_path)
