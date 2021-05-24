###############################################################
# NATS-Bench (arxiv.org/pdf/2009.00437.pdf), IEEE TPAMI 2021  #
# The code to draw Figure 2 / 3 / 4 / 5 in our paper.         #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-ranks.py                 #
###############################################################
import os, sys, time, torch, argparse
import scipy
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
from xautodl.models import get_cell_based_tiny_net
from nats_bench import create


name2label = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "ImageNet16-120": "ImageNet-16-120",
}


def visualize_relative_info(vis_save_dir, search_space, indicator, topk):
    vis_save_dir = vis_save_dir.resolve()
    print(
        "{:} start to visualize {:} with top-{:} information".format(
            time_string(), search_space, topk
        )
    )
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    cache_file_path = vis_save_dir / "cache-{:}-info.pth".format(search_space)
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    if not cache_file_path.exists():
        api = create(None, search_space, fast_mode=False, verbose=False)
        all_infos = OrderedDict()
        for index in range(len(api)):
            all_info = OrderedDict()
            for dataset in datasets:
                info_less = api.get_more_info(index, dataset, hp="12", is_random=False)
                info_more = api.get_more_info(
                    index, dataset, hp=api.full_train_epochs, is_random=False
                )
                all_info[dataset] = dict(
                    less=info_less["test-accuracy"], more=info_more["test-accuracy"]
                )
            all_infos[index] = all_info
        torch.save(all_infos, cache_file_path)
        print("{:} save all cache data into {:}".format(time_string(), cache_file_path))
    else:
        api = create(None, search_space, fast_mode=True, verbose=False)
        all_infos = torch.load(cache_file_path)

    dpi, width, height = 250, 5000, 1300
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 16, 16

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]

    def sub_plot_fn(ax, dataset, indicator):
        performances = []
        # pickup top 10% architectures
        for _index in range(len(api)):
            performances.append((all_infos[_index][dataset][indicator], _index))
        performances = sorted(performances, reverse=True)
        performances = performances[: int(len(api) * topk * 0.01)]
        selected_indexes = [x[1] for x in performances]
        print(
            "{:} plot {:10s} with {:}, {:} architectures".format(
                time_string(), dataset, indicator, len(selected_indexes)
            )
        )
        standard_scores = []
        random_scores = []
        for idx in selected_indexes:
            standard_scores.append(
                api.get_more_info(
                    idx,
                    dataset,
                    hp=api.full_train_epochs if indicator == "more" else "12",
                    is_random=False,
                )["test-accuracy"]
            )
            random_scores.append(
                api.get_more_info(
                    idx,
                    dataset,
                    hp=api.full_train_epochs if indicator == "more" else "12",
                    is_random=True,
                )["test-accuracy"]
            )
        indexes = list(range(len(selected_indexes)))
        standard_indexes = sorted(indexes, key=lambda i: standard_scores[i])
        random_indexes = sorted(indexes, key=lambda i: random_scores[i])
        random_labels = []
        for idx in standard_indexes:
            random_labels.append(random_indexes.index(idx))
        for tick in ax.get_xticklabels():
            tick.set_fontsize(LabelSize - 3)
        for tick in ax.get_yticklabels():
            tick.set_rotation(25)
            tick.set_fontsize(LabelSize - 3)
        ax.set_xlim(0, len(indexes))
        ax.set_ylim(0, len(indexes))
        ax.set_yticks(np.arange(min(indexes), max(indexes), max(indexes) // 3))
        ax.set_xticks(np.arange(min(indexes), max(indexes), max(indexes) // 5))
        ax.scatter(indexes, random_labels, marker="^", s=0.5, c="tab:green", alpha=0.8)
        ax.scatter(indexes, indexes, marker="o", s=0.5, c="tab:blue", alpha=0.8)
        ax.scatter(
            [-1],
            [-1],
            marker="o",
            s=100,
            c="tab:blue",
            label="Average Over Multi-Trials",
        )
        ax.scatter(
            [-1],
            [-1],
            marker="^",
            s=100,
            c="tab:green",
            label="Randomly Selected Trial",
        )

        coef, p = scipy.stats.kendalltau(standard_scores, random_scores)
        ax.set_xlabel(
            "architecture ranking in {:}".format(name2label[dataset]),
            fontsize=LabelSize,
        )
        if dataset == "cifar10":
            ax.set_ylabel("architecture ranking", fontsize=LabelSize)
        ax.legend(loc=4, fontsize=LegendFontsize)
        return coef

    for dataset, ax in zip(datasets, axs):
        rank_coef = sub_plot_fn(ax, dataset, indicator)
        print(
            "sub-plot {:} on {:} done, the ranking coefficient is {:.4f}.".format(
                dataset, search_space, rank_coef
            )
        )

    save_path = (
        vis_save_dir / "{:}-rank-{:}-top{:}.pdf".format(search_space, indicator, topk)
    ).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (
        vis_save_dir / "{:}-rank-{:}-top{:}.png".format(search_space, indicator, topk)
    ).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("Save into {:}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NATS-Bench", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/vis-nas-bench/rank-stability",
        help="Folder to save checkpoints and log.",
    )
    args = parser.parse_args()
    to_save_dir = Path(args.save_dir)

    for topk in [1, 5, 10, 20]:
        visualize_relative_info(to_save_dir, "tss", "more", topk)
        visualize_relative_info(to_save_dir, "sss", "less", topk)
    print("{:} : complete running this file : {:}".format(time_string(), __file__))
