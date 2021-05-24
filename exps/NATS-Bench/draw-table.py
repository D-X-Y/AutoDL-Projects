###############################################################
# NATS-Bench (arxiv.org/pdf/2009.00437.pdf), IEEE TPAMI 2021  #
# The code to draw some results in Table 4 in our paper.      #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-table.py                 #
###############################################################
import os, gc, sys, time, torch, argparse
import numpy as np
from typing import List, Text, Dict, Any
from shutil import copyfile
from collections import defaultdict, OrderedDict
from copy import deepcopy
from pathlib import Path
import matplotlib
import seaborn as sns

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from xautodl.config_utils import dict2config, load_config
from xautodl.log_utils import time_string
from nats_bench import create


def fetch_data(root_dir="./output/search", search_space="tss", dataset=None):
    ss_dir = "{:}-{:}".format(root_dir, search_space)
    alg2name, alg2path = OrderedDict(), OrderedDict()
    alg2name["REA"] = "R-EA-SS3"
    alg2name["REINFORCE"] = "REINFORCE-0.01"
    alg2name["RANDOM"] = "RANDOM"
    alg2name["BOHB"] = "BOHB"
    for alg, name in alg2name.items():
        alg2path[alg] = os.path.join(ss_dir, dataset, name, "results.pth")
        assert os.path.isfile(alg2path[alg]), "invalid path : {:}".format(alg2path[alg])
    alg2data = OrderedDict()
    for alg, path in alg2path.items():
        data = torch.load(path)
        for index, info in data.items():
            info["time_w_arch"] = [
                (x, y) for x, y in zip(info["all_total_times"], info["all_archs"])
            ]
            for j, arch in enumerate(info["all_archs"]):
                assert arch != -1, "invalid arch from {:} {:} {:} ({:}, {:})".format(
                    alg, search_space, dataset, index, j
                )
        alg2data[alg] = data
    return alg2data


def get_valid_test_acc(api, arch, dataset):
    is_size_space = api.search_space_name == "size"
    if dataset == "cifar10":
        xinfo = api.get_more_info(
            arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
        )
        test_acc = xinfo["test-accuracy"]
        xinfo = api.get_more_info(
            arch,
            dataset="cifar10-valid",
            hp=90 if is_size_space else 200,
            is_random=False,
        )
        valid_acc = xinfo["valid-accuracy"]
    else:
        xinfo = api.get_more_info(
            arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
        )
        valid_acc = xinfo["valid-accuracy"]
        test_acc = xinfo["test-accuracy"]
    return (
        valid_acc,
        test_acc,
        "validation = {:.2f}, test = {:.2f}\n".format(valid_acc, test_acc),
    )


def show_valid_test(api, arch):
    is_size_space = api.search_space_name == "size"
    final_str = ""
    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        valid_acc, test_acc, perf_str = get_valid_test_acc(api, arch, dataset)
        final_str += "{:} : {:}\n".format(dataset, perf_str)
    return final_str


def find_best_valid(api, dataset):
    all_valid_accs, all_test_accs = [], []
    for index, arch in enumerate(api):
        valid_acc, test_acc, perf_str = get_valid_test_acc(api, index, dataset)
        all_valid_accs.append((index, valid_acc))
        all_test_accs.append((index, test_acc))
    best_valid_index = sorted(all_valid_accs, key=lambda x: -x[1])[0][0]
    best_test_index = sorted(all_test_accs, key=lambda x: -x[1])[0][0]

    print("-" * 50 + "{:10s}".format(dataset) + "-" * 50)
    print(
        "Best ({:}) architecture on validation: {:}".format(
            best_valid_index, api[best_valid_index]
        )
    )
    print(
        "Best ({:}) architecture on       test: {:}".format(
            best_test_index, api[best_test_index]
        )
    )
    _, _, perf_str = get_valid_test_acc(api, best_valid_index, dataset)
    print("using validation ::: {:}".format(perf_str))
    _, _, perf_str = get_valid_test_acc(api, best_test_index, dataset)
    print("using test       ::: {:}".format(perf_str))


def interplate_fn(xpair1, xpair2, x):
    (x1, y1) = xpair1
    (x2, y2) = xpair2
    return (x2 - x) / (x2 - x1) * y1 + (x - x1) / (x2 - x1) * y2


def query_performance(api, info, dataset, ticket):
    info = deepcopy(info)
    results, is_size_space = [], api.search_space_name == "size"
    time_w_arch = sorted(info["time_w_arch"], key=lambda x: abs(x[0] - ticket))
    time_a, arch_a = time_w_arch[0]
    time_b, arch_b = time_w_arch[1]

    v_acc_a, t_acc_a, _ = get_valid_test_acc(api, arch_a, dataset)
    v_acc_b, t_acc_b, _ = get_valid_test_acc(api, arch_b, dataset)
    v_acc = interplate_fn((time_a, v_acc_a), (time_b, v_acc_b), ticket)
    t_acc = interplate_fn((time_a, t_acc_a), (time_b, t_acc_b), ticket)
    # if True:
    #   interplate = (time_b-ticket) / (time_b-time_a) * accuracy_a + (ticket-time_a) / (time_b-time_a) * accuracy_b
    #   results.append(interplate)
    # return sum(results) / len(results)
    return v_acc, t_acc


def show_multi_trial(search_space):
    api = create(None, search_space, fast_mode=True, verbose=False)

    def show(dataset):
        print("show {:} on {:} done.".format(dataset, search_space))
        xdataset, max_time = dataset.split("-T")
        alg2data = fetch_data(search_space=search_space, dataset=dataset)
        for idx, (alg, data) in enumerate(alg2data.items()):

            valid_accs, test_accs = [], []
            for _, x in data.items():
                v_acc, t_acc = query_performance(api, x, xdataset, float(max_time))
                valid_accs.append(v_acc)
                test_accs.append(t_acc)
            valid_str = "{:.2f}$\pm${:.2f}".format(
                np.mean(valid_accs), np.std(valid_accs)
            )
            test_str = "{:.2f}$\pm${:.2f}".format(np.mean(test_accs), np.std(test_accs))
            print(
                "{:} plot alg : {:10s}  | validation = {:} | test = {:}".format(
                    time_string(), alg, valid_str, test_str
                )
            )

    if search_space == "tss":
        datasets = ["cifar10-T20000", "cifar100-T40000", "ImageNet16-120-T120000"]
    elif search_space == "sss":
        datasets = ["cifar10-T20000", "cifar100-T40000", "ImageNet16-120-T60000"]
    else:
        raise ValueError("Unknown search space: {:}".format(search_space))
    for dataset in datasets:
        show(dataset)
    print("{:} complete show multi-trial results.\n".format(time_string()))


if __name__ == "__main__":

    show_multi_trial("tss")
    show_multi_trial("sss")

    api_tss = create(None, "tss", fast_mode=False, verbose=False)
    resnet = "|nor_conv_3x3~0|+|none~0|nor_conv_3x3~1|+|skip_connect~0|none~1|skip_connect~2|"
    resnet_index = api_tss.query_index_by_arch(resnet)
    print(show_valid_test(api_tss, resnet_index))

    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        find_best_valid(api_tss, dataset)

    largest = "64:64:64:64:64"
    largest_index = api_sss.query_index_by_arch(largest)
    print(show_valid_test(api_sss, largest_index))
    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        find_best_valid(api_sss, dataset)
