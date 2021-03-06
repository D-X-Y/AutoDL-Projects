#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
# python exps/trading/organize_results.py
#####################################################
import sys, argparse
import numpy as np
from typing import List, Text
from collections import defaultdict, OrderedDict
from pathlib import Path
from pprint import pprint
import ruamel.yaml as yaml

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import qlib
from qlib.config import REG_CN
from qlib.workflow import R


class QResult:
    def __init__(self):
        self._result = defaultdict(list)

    def append(self, key, value):
        self._result[key].append(value)

    @property
    def result(self):
        return self._result

    def update(self, metrics, filter_keys=None):
        for key, value in metrics.items():
            if filter_keys is not None and key in filter_keys:
                key = filter_keys[key]
            elif filter_keys is not None:
                continue
            self.append(key, value)

    @staticmethod
    def full_str(xstr, space):
        xformat = "{:" + str(space) + "s}"
        return xformat.format(str(xstr))

    def info(self, keys: List[Text], separate: Text = "", space: int = 25, show=True):
        avaliable_keys = []
        for key in keys:
            if key not in self.result:
                print("There are invalid key [{:}].".format(key))
            else:
                avaliable_keys.append(key)
        head_str = separate.join([self.full_str(x, space) for x in avaliable_keys])
        values = []
        for key in avaliable_keys:
            current_values = self._result[key]
            mean = np.mean(current_values)
            std = np.std(current_values)
            values.append("{:.4f} $\pm$ {:.4f}".format(mean, std))
        value_str = separate.join([self.full_str(x, space) for x in values])
        if show:
            print(head_str)
            print(value_str)
        else:
            return head_str, value_str


def compare_results(heads, values, names, space=10):
    for idx, x in enumerate(heads):
        assert x == heads[0], "[{:}] {:} vs {:}".format(idx, x, heads[0])
    new_head = QResult.full_str("Name", space) + heads[0]
    print(new_head)
    for name, value in zip(names, values):
        xline = QResult.full_str(name, space) + value
        print(xline)


def filter_finished(recorders):
    returned_recorders = dict()
    not_finished = 0
    for key, recorder in recorders.items():
        if recorder.status == "FINISHED":
            returned_recorders[key] = recorder
        else:
            not_finished += 1
    return returned_recorders, not_finished


def main(xargs):
    R.reset_default_uri(xargs.save_dir)
    experiments = R.list_experiments()

    key_map = {
        "IC": "IC",
        "ICIR": "ICIR",
        "Rank IC": "Rank_IC",
        "Rank ICIR": "Rank_ICIR",
        "excess_return_with_cost.annualized_return": "Annualized_Return",
        "excess_return_with_cost.information_ratio": "Information_Ratio",
        "excess_return_with_cost.max_drawdown": "Max_Drawdown",
    }
    all_keys = list(key_map.values())

    print("There are {:} experiments.".format(len(experiments)))
    head_strs, value_strs, names = [], [], []
    for idx, (key, experiment) in enumerate(experiments.items()):
        if experiment.id == "0":
            continue
        recorders = experiment.list_recorders()
        recorders, not_finished = filter_finished(recorders)
        print(
            "====>>>> {:02d}/{:02d}-th experiment {:9s} has {:02d}/{:02d} finished recorders.".format(
                idx, len(experiments), experiment.name, len(recorders), len(recorders) + not_finished
            )
        )
        result = QResult()
        for recorder_id, recorder in recorders.items():
            result.update(recorder.list_metrics(), key_map)
        head_str, value_str = result.info(all_keys, show=False)
        head_strs.append(head_str)
        value_strs.append(value_str)
        names.append(experiment.name)
    compare_results(head_strs, value_strs, names, space=10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Show Results")
    parser.add_argument("--save_dir", type=str, default="./outputs/qlib-baselines", help="The checkpoint directory.")
    args = parser.parse_args()

    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    main(args)
