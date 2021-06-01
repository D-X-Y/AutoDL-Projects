#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
# python exps/trading/organize_results.py --save_dir outputs/qlib-baselines-all
#####################################################
import os, re, sys, argparse
import numpy as np
from typing import List, Text
from collections import defaultdict, OrderedDict
from pprint import pprint
from pathlib import Path
import ruamel.yaml as yaml

lib_dir = (Path(__file__).parent / ".." / "..").resolve()
print("LIB-DIR: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from xautodl.config_utils import arg_str2bool
from xautodl.utils.qlib_utils import QResult

import qlib
from qlib.config import REG_CN
from qlib.workflow import R


def compare_results(
    heads, values, names, space=10, separate="& ", verbose=True, sort_key=False
):
    for idx, x in enumerate(heads):
        assert x == heads[0], "[{:}] \n{:}\nvs\n{:}".format(idx, x, heads[0])
    new_head = QResult.full_str("Name", space) + separate + heads[0]
    info_str_dict = dict(head=new_head, lines=[])
    for name, value in zip(names, values):
        xline = QResult.full_str(name, space) + separate + value
        info_str_dict["lines"].append(xline)
    if verbose:
        print("\nThere are {:} algorithms.".format(len(values)))
        print(info_str_dict["head"])
        if sort_key:
            lines = sorted(
                list(zip(values, info_str_dict["lines"])),
                key=lambda x: float(x[0].split(" ")[0]),
            )
            lines = [x[1] for x in lines]
        else:
            lines = info_str_dict["lines"]
        for xline in lines:
            print(xline + "\\\\")
    return info_str_dict


def filter_finished(recorders):
    returned_recorders = dict()
    not_finished = 0
    for key, recorder in recorders.items():
        if recorder.status == "FINISHED":
            returned_recorders[key] = recorder
        else:
            not_finished += 1
    return returned_recorders, not_finished


def query_info(save_dir, verbose, name_filter, key_map):
    R.set_uri(save_dir)
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
            result.update(recorder.list_metrics(), key_map)
            result.append_path(
                os.path.join(recorder.uri, recorder.experiment_id, recorder.id)
            )
        if not len(result):
            print("There are no valid recorders for {:}".format(experiment))
            continue
        else:
            print(
                "There are {:} valid recorders for {:}".format(
                    len(recorders), experiment.name
                )
            )
        qresults.append(result)
    return qresults


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Show Results")

    parser.add_argument(
        "--save_dir",
        type=str,
        nargs="+",
        default=[],
        help="The checkpoint directory.",
    )
    parser.add_argument(
        "--verbose",
        type=arg_str2bool,
        default=False,
        help="Print detailed log information or not.",
    )
    parser.add_argument(
        "--name_filter", type=str, default=".*", help="Filter experiment names."
    )
    args = parser.parse_args()

    print("Show results of {:}".format(args.save_dir))
    if not args.save_dir:
        raise ValueError("Receive no input directory for [args.save_dir]")

    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    """
    key_map = {
        # "RMSE": "RMSE",
        "IC": "IC",
        "ICIR": "ICIR",
        "Rank IC": "Rank_IC",
        "Rank ICIR": "Rank_ICIR",
        # "excess_return_with_cost.annualized_return": "Annualized_Return",
        # "excess_return_with_cost.information_ratio": "Information_Ratio",
        # "excess_return_with_cost.max_drawdown": "Max_Drawdown",
    }
    """
    key_map = dict()
    for xset in ("train", "valid", "test"):
        key_map["{:}-mean-IC".format(xset)] = "IC ({:})".format(xset)
        # key_map["{:}-mean-ICIR".format(xset)] = "ICIR ({:})".format(xset)
        key_map["{:}-mean-Rank-IC".format(xset)] = "Rank IC ({:})".format(xset)
        # key_map["{:}-mean-Rank-ICIR".format(xset)] = "Rank ICIR ({:})".format(xset)

    all_qresults = []
    for save_dir in args.save_dir:
        qresults = query_info(save_dir, args.verbose, args.name_filter, key_map)
        all_qresults.extend(qresults)
    names, head_strs, value_strs = [], [], []
    for result in all_qresults:
        head_str, value_str = result.info(list(key_map.values()), verbose=args.verbose)
        head_strs.append(head_str)
        value_strs.append(value_str)
        names.append(result.name)
    compare_results(
        head_strs,
        value_strs,
        names,
        space=18,
        verbose=True,
        sort_key=True,
    )
