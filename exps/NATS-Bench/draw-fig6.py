###############################################################
# NATS-Bench (arxiv.org/pdf/2009.00437.pdf), IEEE TPAMI 2021  #
# The code to draw Figure 6 in our paper.                     #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-fig6.py --search_space tss
# Usage: python exps/NATS-Bench/draw-fig6.py --search_space sss
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


def show_valid_test(api, data, dataset):
    valid_accs, test_accs, is_size_space = [], [], api.search_space_name == "size"
    for i, info in data.items():
        time, arch = info["time_w_arch"][-1]
        if dataset == "cifar10":
            xinfo = api.get_more_info(
                arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
            )
            test_accs.append(xinfo["test-accuracy"])
            xinfo = api.get_more_info(
                arch,
                dataset="cifar10-valid",
                hp=90 if is_size_space else 200,
                is_random=False,
            )
            valid_accs.append(xinfo["valid-accuracy"])
        else:
            xinfo = api.get_more_info(
                arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
            )
            valid_accs.append(xinfo["valid-accuracy"])
            test_accs.append(xinfo["test-accuracy"])
    valid_str = "{:.2f}$\pm${:.2f}".format(np.mean(valid_accs), np.std(valid_accs))
    test_str = "{:.2f}$\pm${:.2f}".format(np.mean(test_accs), np.std(test_accs))
    return valid_str, test_str


y_min_s = {
    ("cifar10", "tss"): 90,
    ("cifar10", "sss"): 92,
    ("cifar100", "tss"): 65,
    ("cifar100", "sss"): 65,
    ("ImageNet16-120", "tss"): 36,
    ("ImageNet16-120", "sss"): 40,
}

y_max_s = {
    ("cifar10", "tss"): 94.3,
    ("cifar10", "sss"): 93.3,
    ("cifar100", "tss"): 72.5,
    ("cifar100", "sss"): 70.5,
    ("ImageNet16-120", "tss"): 46,
    ("ImageNet16-120", "sss"): 46,
}

x_axis_s = {
    ("cifar10", "tss"): 200,
    ("cifar10", "sss"): 200,
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


def visualize_curve(api, vis_save_dir, search_space):
    vis_save_dir = vis_save_dir.resolve()
    vis_save_dir.mkdir(parents=True, exist_ok=True)

    dpi, width, height = 250, 5200, 1400
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 16, 16

    def sub_plot_fn(ax, dataset):
        xdataset, max_time = dataset.split("-T")
        alg2data = fetch_data(search_space=search_space, dataset=dataset)
        alg2accuracies = OrderedDict()
        total_tickets = 150
        time_tickets = [
            float(i) / total_tickets * int(max_time) for i in range(total_tickets)
        ]
        colors = ["b", "g", "c", "m", "y"]
        ax.set_xlim(0, x_axis_s[(xdataset, search_space)])
        ax.set_ylim(
            y_min_s[(xdataset, search_space)], y_max_s[(xdataset, search_space)]
        )
        for idx, (alg, data) in enumerate(alg2data.items()):
            accuracies = []
            for ticket in time_tickets:
                accuracy, accuracy_std = query_performance(api, data, xdataset, ticket)
                accuracies.append(accuracy)
            valid_str, test_str = show_valid_test(api, data, xdataset)
            # print('{:} plot alg : {:10s}, final accuracy = {:.2f}$\pm${:.2f}'.format(time_string(), alg, accuracy, accuracy_std))
            print(
                "{:} plot alg : {:10s}  | validation = {:} | test = {:}".format(
                    time_string(), alg, valid_str, test_str
                )
            )
            alg2accuracies[alg] = accuracies
            ax.plot(
                [x / 100 for x in time_tickets],
                accuracies,
                c=colors[idx],
                label="{:}".format(alg),
            )
            ax.set_xlabel("Estimated wall-clock time (1e2 seconds)", fontsize=LabelSize)
            ax.set_ylabel(
                "Test accuracy on {:}".format(name2label[xdataset]), fontsize=LabelSize
            )
            ax.set_title(
                "Searching results on {:}".format(name2label[xdataset]),
                fontsize=LabelSize + 4,
            )
        ax.legend(loc=4, fontsize=LegendFontsize)

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    # datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
    if search_space == "tss":
        datasets = ["cifar10-T20000", "cifar100-T40000", "ImageNet16-120-T120000"]
    elif search_space == "sss":
        datasets = ["cifar10-T20000", "cifar100-T40000", "ImageNet16-120-T60000"]
    else:
        raise ValueError("Unknown search space: {:}".format(search_space))
    for dataset, ax in zip(datasets, axs):
        sub_plot_fn(ax, dataset)
        print("sub-plot {:} on {:} done.".format(dataset, search_space))
    save_path = (vis_save_dir / "{:}-curve.png".format(search_space)).resolve()
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

    api = create(None, args.search_space, fast_mode=True, verbose=False)
    visualize_curve(api, save_dir, args.search_space)
