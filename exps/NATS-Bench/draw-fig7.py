###############################################################
# NATS-Bench (arxiv.org/pdf/2009.00437.pdf), IEEE TPAMI 2021  #
# The code to draw Figure 7 in our paper.                     #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/NATS-Bench/draw-fig7.py                  #
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


def fetch_data(
    root_dir="./output/search", search_space="tss", dataset=None, suffix="-WARM0.3"
):
    ss_dir = "{:}-{:}".format(root_dir, search_space)
    alg2name, alg2path = OrderedDict(), OrderedDict()
    seeds = [777, 888, 999]
    print("\n[fetch data] from {:} on {:}".format(search_space, dataset))
    if search_space == "tss":
        alg2name["GDAS"] = "gdas-affine0_BN0-None"
        alg2name["RSPS"] = "random-affine0_BN0-None"
        alg2name["DARTS (1st)"] = "darts-v1-affine0_BN0-None"
        alg2name["DARTS (2nd)"] = "darts-v2-affine0_BN0-None"
        alg2name["ENAS"] = "enas-affine0_BN0-None"
        alg2name["SETN"] = "setn-affine0_BN0-None"
    else:
        alg2name["channel-wise interpolation"] = "tas-affine0_BN0-AWD0.001{:}".format(
            suffix
        )
        alg2name[
            "masking + Gumbel-Softmax"
        ] = "mask_gumbel-affine0_BN0-AWD0.001{:}".format(suffix)
        alg2name["masking + sampling"] = "mask_rl-affine0_BN0-AWD0.0{:}".format(suffix)
    for alg, name in alg2name.items():
        alg2path[alg] = os.path.join(ss_dir, dataset, name, "seed-{:}-last-info.pth")
    alg2data = OrderedDict()
    for alg, path in alg2path.items():
        alg2data[alg], ok_num = [], 0
        for seed in seeds:
            xpath = path.format(seed)
            if os.path.isfile(xpath):
                ok_num += 1
            else:
                print("This is an invalid path : {:}".format(xpath))
                continue
            data = torch.load(xpath, map_location=torch.device("cpu"))
            try:
                data = torch.load(
                    data["last_checkpoint"], map_location=torch.device("cpu")
                )
            except:
                xpath = str(data["last_checkpoint"]).split("E100-")
                if len(xpath) == 2 and os.path.isfile(xpath[0] + xpath[1]):
                    xpath = xpath[0] + xpath[1]
                elif "fbv2" in str(data["last_checkpoint"]):
                    xpath = str(data["last_checkpoint"]).replace("fbv2", "mask_gumbel")
                elif "tunas" in str(data["last_checkpoint"]):
                    xpath = str(data["last_checkpoint"]).replace("tunas", "mask_rl")
                else:
                    raise ValueError(
                        "Invalid path: {:}".format(data["last_checkpoint"])
                    )
                data = torch.load(xpath, map_location=torch.device("cpu"))
            alg2data[alg].append(data["genotypes"])
        print("This algorithm : {:} has {:} valid ckps.".format(alg, ok_num))
        assert ok_num > 0, "Must have at least 1 valid ckps."
    return alg2data


y_min_s = {
    ("cifar10", "tss"): 90,
    ("cifar10", "sss"): 92,
    ("cifar100", "tss"): 65,
    ("cifar100", "sss"): 65,
    ("ImageNet16-120", "tss"): 36,
    ("ImageNet16-120", "sss"): 40,
}

y_max_s = {
    ("cifar10", "tss"): 94.5,
    ("cifar10", "sss"): 93.3,
    ("cifar100", "tss"): 72,
    ("cifar100", "sss"): 70,
    ("ImageNet16-120", "tss"): 44,
    ("ImageNet16-120", "sss"): 46,
}

name2label = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "ImageNet16-120": "ImageNet-16-120",
}

name2suffix = {
    ("sss", "warm"): "-WARM0.3",
    ("sss", "none"): "-WARMNone",
    ("tss", "none"): None,
    ("tss", None): None,
}


def visualize_curve(api, vis_save_dir, search_space, suffix):
    vis_save_dir = vis_save_dir.resolve()
    vis_save_dir.mkdir(parents=True, exist_ok=True)

    dpi, width, height = 250, 5200, 1400
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 16, 16

    def sub_plot_fn(ax, dataset):
        print("{:} plot {:10s}".format(time_string(), dataset))
        alg2data = fetch_data(
            search_space=search_space,
            dataset=dataset,
            suffix=name2suffix[(search_space, suffix)],
        )
        alg2accuracies = OrderedDict()
        epochs = 100
        colors = ["b", "g", "c", "m", "y", "r"]
        ax.set_xlim(0, epochs)
        # ax.set_ylim(y_min_s[(dataset, search_space)], y_max_s[(dataset, search_space)])
        for idx, (alg, data) in enumerate(alg2data.items()):
            xs, accuracies = [], []
            for iepoch in range(epochs + 1):
                try:
                    structures, accs = [_[iepoch - 1] for _ in data], []
                except:
                    raise ValueError(
                        "This alg {:} on {:} has invalid checkpoints.".format(
                            alg, dataset
                        )
                    )
                for structure in structures:
                    info = api.get_more_info(
                        structure,
                        dataset=dataset,
                        hp=90 if api.search_space_name == "size" else 200,
                        is_random=False,
                    )
                    accs.append(info["test-accuracy"])
                accuracies.append(sum(accs) / len(accs))
                xs.append(iepoch)
            alg2accuracies[alg] = accuracies
            ax.plot(xs, accuracies, c=colors[idx], label="{:}".format(alg))
            ax.set_xlabel("The searching epoch", fontsize=LabelSize)
            ax.set_ylabel(
                "Test accuracy on {:}".format(name2label[dataset]), fontsize=LabelSize
            )
            ax.set_title(
                "Searching results on {:}".format(name2label[dataset]),
                fontsize=LabelSize + 4,
            )
            structures, valid_accs, test_accs = [_[epochs - 1] for _ in data], [], []
            print(
                "{:} plot alg : {:} -- final {:} architectures.".format(
                    time_string(), alg, len(structures)
                )
            )
            for arch in structures:
                valid_acc, test_acc, _ = get_valid_test_acc(api, arch, dataset)
                test_accs.append(test_acc)
                valid_accs.append(valid_acc)
            print(
                "{:} plot alg : {:} -- validation: {:.2f}$\pm${:.2f} -- test: {:.2f}$\pm${:.2f}".format(
                    time_string(),
                    alg,
                    np.mean(valid_accs),
                    np.std(valid_accs),
                    np.mean(test_accs),
                    np.std(test_accs),
                )
            )
        ax.legend(loc=4, fontsize=LegendFontsize)

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    for dataset, ax in zip(datasets, axs):
        sub_plot_fn(ax, dataset)
        print("sub-plot {:} on {:} done.".format(dataset, search_space))
    save_path = (
        vis_save_dir / "{:}-ws-{:}-curve.png".format(search_space, suffix)
    ).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NATS-Bench", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/vis-nas-bench/nas-algos",
        help="Folder to save checkpoints and log.",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    api_tss = create(None, "tss", fast_mode=True, verbose=False)
    visualize_curve(api_tss, save_dir, "tss", None)

    api_sss = create(None, "sss", fast_mode=True, verbose=False)
    visualize_curve(api_sss, save_dir, "sss", "warm")
    visualize_curve(api_sss, save_dir, "sss", "none")
