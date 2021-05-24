###############################################################
# NATS-Bench (arxiv.org/pdf/2009.00437.pdf), IEEE TPAMI 2021  #
# The code to draw Figure 6 in our paper.                     #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-fig8.py                  #
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


plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)
## for Palatino and other serif fonts use:
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)


def fetch_data(root_dir="./output/search", search_space="tss", dataset=None):
    ss_dir = "{:}-{:}".format(root_dir, search_space)
    alg2all = OrderedDict()
    # alg2name['REINFORCE'] = 'REINFORCE-0.01'
    # alg2name['RANDOM'] = 'RANDOM'
    # alg2name['BOHB'] = 'BOHB'
    if search_space == "tss":
        hp = "$\mathcal{H}^{1}$"
        if dataset == "cifar10":
            suffixes = ["-T1200000", "-T1200000-FULL"]
    elif search_space == "sss":
        hp = "$\mathcal{H}^{2}$"
        if dataset == "cifar10":
            suffixes = ["-T200000", "-T200000-FULL"]
    else:
        raise ValueError("Unkonwn search space: {:}".format(search_space))

    alg2all[r"REA ($\mathcal{H}^{0}$)"] = dict(
        path=os.path.join(ss_dir, dataset + suffixes[0], "R-EA-SS3", "results.pth"),
        color="b",
        linestyle="-",
    )
    alg2all[r"REA ({:})".format(hp)] = dict(
        path=os.path.join(ss_dir, dataset + suffixes[1], "R-EA-SS3", "results.pth"),
        color="b",
        linestyle="--",
    )

    for alg, xdata in alg2all.items():
        data = torch.load(xdata["path"])
        for index, info in data.items():
            info["time_w_arch"] = [
                (x, y) for x, y in zip(info["all_total_times"], info["all_archs"])
            ]
            for j, arch in enumerate(info["all_archs"]):
                assert arch != -1, "invalid arch from {:} {:} {:} ({:}, {:})".format(
                    alg, search_space, dataset, index, j
                )
        xdata["data"] = data
    return alg2all


def query_performance(api, data, dataset, ticket):
    results, is_size_space = [], api.search_space_name == "size"
    for i, info in data.items():
        time_w_arch = sorted(info["time_w_arch"], key=lambda x: abs(x[0] - ticket))
        time_a, arch_a = time_w_arch[0]
        time_b, arch_b = time_w_arch[1]
        info_a = api.get_more_info(
            arch_a, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
        )
        info_b = api.get_more_info(
            arch_b, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
        )
        accuracy_a, accuracy_b = info_a["test-accuracy"], info_b["test-accuracy"]
        interplate = (time_b - ticket) / (time_b - time_a) * accuracy_a + (
            ticket - time_a
        ) / (time_b - time_a) * accuracy_b
        results.append(interplate)
    # return sum(results) / len(results)
    return np.mean(results), np.std(results)


y_min_s = {
    ("cifar10", "tss"): 91,
    ("cifar10", "sss"): 91,
    ("cifar100", "tss"): 65,
    ("cifar100", "sss"): 65,
    ("ImageNet16-120", "tss"): 36,
    ("ImageNet16-120", "sss"): 40,
}

y_max_s = {
    ("cifar10", "tss"): 94.5,
    ("cifar10", "sss"): 93.5,
    ("cifar100", "tss"): 72.5,
    ("cifar100", "sss"): 70.5,
    ("ImageNet16-120", "tss"): 46,
    ("ImageNet16-120", "sss"): 46,
}

x_axis_s = {
    ("cifar10", "tss"): 1200000,
    ("cifar10", "sss"): 200000,
    ("cifar100", "tss"): 400,
    ("cifar100", "sss"): 400,
    ("ImageNet16-120", "tss"): 1200,
    ("ImageNet16-120", "sss"): 600,
}

name2label = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "ImageNet16-120": "ImageNet-16-120",
}

spaces2latex = {
    "tss": r"$\mathcal{S}_{t}$",
    "sss": r"$\mathcal{S}_{s}$",
}


# FuncFormatter can be used as a decorator
@ticker.FuncFormatter
def major_formatter(x, pos):
    if x == 0:
        return "0"
    else:
        return "{:.2f}e5".format(x / 1e5)


def visualize_curve(api_dict, vis_save_dir):
    vis_save_dir = vis_save_dir.resolve()
    vis_save_dir.mkdir(parents=True, exist_ok=True)

    dpi, width, height = 250, 5000, 2000
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 28, 28

    def sub_plot_fn(ax, search_space, dataset):
        max_time = x_axis_s[(dataset, search_space)]
        alg2data = fetch_data(search_space=search_space, dataset=dataset)
        alg2accuracies = OrderedDict()
        total_tickets = 200
        time_tickets = [
            float(i) / total_tickets * int(max_time) for i in range(total_tickets)
        ]
        ax.set_xlim(0, x_axis_s[(dataset, search_space)])
        ax.set_ylim(y_min_s[(dataset, search_space)], y_max_s[(dataset, search_space)])
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_fontsize(LabelSize - 6)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(LabelSize - 6)
        ax.xaxis.set_major_formatter(major_formatter)
        for idx, (alg, xdata) in enumerate(alg2data.items()):
            accuracies = []
            for ticket in time_tickets:
                # import pdb; pdb.set_trace()
                accuracy, accuracy_std = query_performance(
                    api_dict[search_space], xdata["data"], dataset, ticket
                )
                accuracies.append(accuracy)
            # print('{:} plot alg : {:10s}, final accuracy = {:.2f}$\pm${:.2f}'.format(time_string(), alg, accuracy, accuracy_std))
            print(
                "{:} plot alg : {:10s} on {:}".format(time_string(), alg, search_space)
            )
            alg2accuracies[alg] = accuracies
            ax.plot(
                time_tickets,
                accuracies,
                c=xdata["color"],
                linestyle=xdata["linestyle"],
                label="{:}".format(alg),
            )
            ax.set_xlabel("Estimated wall-clock time", fontsize=LabelSize)
            ax.set_ylabel("Test accuracy", fontsize=LabelSize)
            ax.set_title(
                r"Results on {:} over {:}".format(
                    name2label[dataset], spaces2latex[search_space]
                ),
                fontsize=LabelSize,
            )
        ax.legend(loc=4, fontsize=LegendFontsize)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    sub_plot_fn(axs[0], "tss", "cifar10")
    sub_plot_fn(axs[1], "sss", "cifar10")
    save_path = (vis_save_dir / "full-curve.png").resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/vis-nas-bench/nas-algos-vs-h",
        help="Folder to save checkpoints and log.",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    api_tss = create(None, "tss", fast_mode=True, verbose=False)
    api_sss = create(None, "sss", fast_mode=True, verbose=False)
    visualize_curve(dict(tss=api_tss, sss=api_sss), save_dir)
