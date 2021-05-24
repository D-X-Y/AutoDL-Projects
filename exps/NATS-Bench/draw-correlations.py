###############################################################
# NATS-Bench (arxiv.org/pdf/2009.00437.pdf), IEEE TPAMI 2021  #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-correlations.py          #
###############################################################
import os, gc, sys, time, scipy, torch, argparse
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


def compute_kendalltau(vectori, vectorj):
    # indexes = list(range(len(vectori)))
    # rank_1 = sorted(indexes, key=lambda i: vectori[i])
    # rank_2 = sorted(indexes, key=lambda i: vectorj[i])
    # import pdb; pdb.set_trace()
    coef, p = scipy.stats.kendalltau(vectori, vectorj)
    return coef


def compute_spearmanr(vectori, vectorj):
    coef, p = scipy.stats.spearmanr(vectori, vectorj)
    return coef


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/vis-nas-bench/nas-algos",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        choices=["tss", "sss"],
        help="Choose the search space.",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    api = create(None, "tss", fast_mode=True, verbose=False)
    indexes = list(range(1, 10000, 300))
    scores_1 = []
    scores_2 = []
    for index in indexes:
        valid_acc, test_acc, _ = get_valid_test_acc(api, index, "cifar10")
        scores_1.append(valid_acc)
        scores_2.append(test_acc)
    correlation = compute_kendalltau(scores_1, scores_2)
    print(
        "The kendall tau correlation of {:} samples : {:}".format(
            len(indexes), correlation
        )
    )
    correlation = compute_spearmanr(scores_1, scores_2)
    print(
        "The spearmanr correlation of {:} samples : {:}".format(
            len(indexes), correlation
        )
    )
    # scores_1 = ['{:.2f}'.format(x) for x in scores_1]
    # scores_2 = ['{:.2f}'.format(x) for x in scores_2]
    # print(', '.join(scores_1))
    # print(', '.join(scores_2))

    dpi, width, height = 250, 1000, 1000
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 14, 14

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(scores_1, scores_2, marker="^", s=0.5, c="tab:green", alpha=0.8)

    save_path = "/Users/xuanyidong/Desktop/test-temp-rank.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    plt.close("all")
