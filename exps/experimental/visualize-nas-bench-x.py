###############################################################
# NAS-Bench-201, ICLR 2020 (https://arxiv.org/abs/2001.00326) #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06           #
###############################################################
# Usage: python exps/experimental/visualize-nas-bench-x.py
###############################################################
import os, sys, time, torch, argparse
import numpy as np
from typing import List, Text, Dict, Any
from shutil import copyfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import matplotlib
import seaborn as sns

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from config_utils import dict2config, load_config
from log_utils import time_string
from models import get_cell_based_tiny_net
from nats_bench import create


def visualize_info(api, vis_save_dir, indicator):
    vis_save_dir = vis_save_dir.resolve()
    # print ('{:} start to visualize {:} information'.format(time_string(), api))
    vis_save_dir.mkdir(parents=True, exist_ok=True)

    cifar010_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "cifar10", indicator
    )
    cifar100_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "cifar100", indicator
    )
    imagenet_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "ImageNet16-120", indicator
    )
    cifar010_info = torch.load(cifar010_cache_path)
    cifar100_info = torch.load(cifar100_cache_path)
    imagenet_info = torch.load(imagenet_cache_path)
    indexes = list(range(len(cifar010_info["params"])))

    print("{:} start to visualize relative ranking".format(time_string()))

    cifar010_ord_indexes = sorted(indexes, key=lambda i: cifar010_info["test_accs"][i])
    cifar100_ord_indexes = sorted(indexes, key=lambda i: cifar100_info["test_accs"][i])
    imagenet_ord_indexes = sorted(indexes, key=lambda i: imagenet_info["test_accs"][i])

    cifar100_labels, imagenet_labels = [], []
    for idx in cifar010_ord_indexes:
        cifar100_labels.append(cifar100_ord_indexes.index(idx))
        imagenet_labels.append(imagenet_ord_indexes.index(idx))
    print("{:} prepare data done.".format(time_string()))

    dpi, width, height = 200, 1400, 800
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 18, 12
    resnet_scale, resnet_alpha = 120, 0.5

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xlim(min(indexes), max(indexes))
    plt.ylim(min(indexes), max(indexes))
    # plt.ylabel('y').set_rotation(30)
    plt.yticks(
        np.arange(min(indexes), max(indexes), max(indexes) // 3),
        fontsize=LegendFontsize,
        rotation="vertical",
    )
    plt.xticks(
        np.arange(min(indexes), max(indexes), max(indexes) // 5),
        fontsize=LegendFontsize,
    )
    ax.scatter(indexes, cifar100_labels, marker="^", s=0.5, c="tab:green", alpha=0.8)
    ax.scatter(indexes, imagenet_labels, marker="*", s=0.5, c="tab:red", alpha=0.8)
    ax.scatter(indexes, indexes, marker="o", s=0.5, c="tab:blue", alpha=0.8)
    ax.scatter([-1], [-1], marker="o", s=100, c="tab:blue", label="CIFAR-10")
    ax.scatter([-1], [-1], marker="^", s=100, c="tab:green", label="CIFAR-100")
    ax.scatter([-1], [-1], marker="*", s=100, c="tab:red", label="ImageNet-16-120")
    plt.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.legend(loc=0, fontsize=LegendFontsize)
    ax.set_xlabel("architecture ranking in CIFAR-10", fontsize=LabelSize)
    ax.set_ylabel("architecture ranking", fontsize=LabelSize)
    save_path = (vis_save_dir / "{:}-relative-rank.pdf".format(indicator)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (vis_save_dir / "{:}-relative-rank.png".format(indicator)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))


def visualize_sss_info(api, dataset, vis_save_dir):
    vis_save_dir = vis_save_dir.resolve()
    print("{:} start to visualize {:} information".format(time_string(), dataset))
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    cache_file_path = vis_save_dir / "{:}-cache-sss-info.pth".format(dataset)
    if not cache_file_path.exists():
        print("Do not find cache file : {:}".format(cache_file_path))
        params, flops, train_accs, valid_accs, test_accs = [], [], [], [], []
        for index in range(len(api)):
            cost_info = api.get_cost_info(index, dataset, hp="90")
            params.append(cost_info["params"])
            flops.append(cost_info["flops"])
            # accuracy
            info = api.get_more_info(index, dataset, hp="90", is_random=False)
            train_accs.append(info["train-accuracy"])
            test_accs.append(info["test-accuracy"])
            if dataset == "cifar10":
                info = api.get_more_info(
                    index, "cifar10-valid", hp="90", is_random=False
                )
                valid_accs.append(info["valid-accuracy"])
            else:
                valid_accs.append(info["valid-accuracy"])
        info = {
            "params": params,
            "flops": flops,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "test_accs": test_accs,
        }
        torch.save(info, cache_file_path)
    else:
        print("Find cache file : {:}".format(cache_file_path))
        info = torch.load(cache_file_path)
        params, flops, train_accs, valid_accs, test_accs = (
            info["params"],
            info["flops"],
            info["train_accs"],
            info["valid_accs"],
            info["test_accs"],
        )
    print("{:} collect data done.".format(time_string()))

    pyramid = [
        "8:16:32:48:64",
        "8:8:16:32:48",
        "8:8:16:16:32",
        "8:8:16:16:48",
        "8:8:16:16:64",
        "16:16:32:32:64",
        "32:32:64:64:64",
    ]
    pyramid_indexes = [api.query_index_by_arch(x) for x in pyramid]
    largest_indexes = [api.query_index_by_arch("64:64:64:64:64")]

    indexes = list(range(len(params)))
    dpi, width, height = 250, 8500, 1300
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 24, 24
    # resnet_scale, resnet_alpha = 120, 0.5
    xscale, xalpha = 120, 0.8

    fig, axs = plt.subplots(1, 4, figsize=figsize)
    # ax1, ax2, ax3, ax4, ax5 = axs
    for ax in axs:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize)
    ax2, ax3, ax4, ax5 = axs
    # ax1.xaxis.set_ticks(np.arange(0, max(indexes), max(indexes)//5))
    # ax1.scatter(indexes, test_accs, marker='o', s=0.5, c='tab:blue')
    # ax1.set_xlabel('architecture ID', fontsize=LabelSize)
    # ax1.set_ylabel('test accuracy (%)', fontsize=LabelSize)

    ax2.scatter(params, train_accs, marker="o", s=0.5, c="tab:blue")
    ax2.scatter(
        [params[x] for x in pyramid_indexes],
        [train_accs[x] for x in pyramid_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="Pyramid Structure",
        alpha=xalpha,
    )
    ax2.scatter(
        [params[x] for x in largest_indexes],
        [train_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax2.set_xlabel("#parameters (MB)", fontsize=LabelSize)
    ax2.set_ylabel("train accuracy (%)", fontsize=LabelSize)
    ax2.legend(loc=4, fontsize=LegendFontsize)

    ax3.scatter(params, test_accs, marker="o", s=0.5, c="tab:blue")
    ax3.scatter(
        [params[x] for x in pyramid_indexes],
        [test_accs[x] for x in pyramid_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="Pyramid Structure",
        alpha=xalpha,
    )
    ax3.scatter(
        [params[x] for x in largest_indexes],
        [test_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax3.set_xlabel("#parameters (MB)", fontsize=LabelSize)
    ax3.set_ylabel("test accuracy (%)", fontsize=LabelSize)
    ax3.legend(loc=4, fontsize=LegendFontsize)

    ax4.scatter(flops, train_accs, marker="o", s=0.5, c="tab:blue")
    ax4.scatter(
        [flops[x] for x in pyramid_indexes],
        [train_accs[x] for x in pyramid_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="Pyramid Structure",
        alpha=xalpha,
    )
    ax4.scatter(
        [flops[x] for x in largest_indexes],
        [train_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax4.set_xlabel("#FLOPs (M)", fontsize=LabelSize)
    ax4.set_ylabel("train accuracy (%)", fontsize=LabelSize)
    ax4.legend(loc=4, fontsize=LegendFontsize)

    ax5.scatter(flops, test_accs, marker="o", s=0.5, c="tab:blue")
    ax5.scatter(
        [flops[x] for x in pyramid_indexes],
        [test_accs[x] for x in pyramid_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="Pyramid Structure",
        alpha=xalpha,
    )
    ax5.scatter(
        [flops[x] for x in largest_indexes],
        [test_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax5.set_xlabel("#FLOPs (M)", fontsize=LabelSize)
    ax5.set_ylabel("test accuracy (%)", fontsize=LabelSize)
    ax5.legend(loc=4, fontsize=LegendFontsize)

    save_path = vis_save_dir / "sss-{:}.png".format(dataset)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


def visualize_tss_info(api, dataset, vis_save_dir):
    vis_save_dir = vis_save_dir.resolve()
    print("{:} start to visualize {:} information".format(time_string(), dataset))
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    cache_file_path = vis_save_dir / "{:}-cache-tss-info.pth".format(dataset)
    if not cache_file_path.exists():
        print("Do not find cache file : {:}".format(cache_file_path))
        params, flops, train_accs, valid_accs, test_accs = [], [], [], [], []
        for index in range(len(api)):
            cost_info = api.get_cost_info(index, dataset, hp="12")
            params.append(cost_info["params"])
            flops.append(cost_info["flops"])
            # accuracy
            info = api.get_more_info(index, dataset, hp="200", is_random=False)
            train_accs.append(info["train-accuracy"])
            test_accs.append(info["test-accuracy"])
            if dataset == "cifar10":
                info = api.get_more_info(
                    index, "cifar10-valid", hp="200", is_random=False
                )
                valid_accs.append(info["valid-accuracy"])
            else:
                valid_accs.append(info["valid-accuracy"])
            print("")
        info = {
            "params": params,
            "flops": flops,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "test_accs": test_accs,
        }
        torch.save(info, cache_file_path)
    else:
        print("Find cache file : {:}".format(cache_file_path))
        info = torch.load(cache_file_path)
        params, flops, train_accs, valid_accs, test_accs = (
            info["params"],
            info["flops"],
            info["train_accs"],
            info["valid_accs"],
            info["test_accs"],
        )
    print("{:} collect data done.".format(time_string()))

    resnet = [
        "|nor_conv_3x3~0|+|none~0|nor_conv_3x3~1|+|skip_connect~0|none~1|skip_connect~2|"
    ]
    resnet_indexes = [api.query_index_by_arch(x) for x in resnet]
    largest_indexes = [
        api.query_index_by_arch(
            "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|"
        )
    ]

    indexes = list(range(len(params)))
    dpi, width, height = 250, 8500, 1300
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 24, 24
    # resnet_scale, resnet_alpha = 120, 0.5
    xscale, xalpha = 120, 0.8

    fig, axs = plt.subplots(1, 4, figsize=figsize)
    # ax1, ax2, ax3, ax4, ax5 = axs
    for ax in axs:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize)
    ax2, ax3, ax4, ax5 = axs
    # ax1.xaxis.set_ticks(np.arange(0, max(indexes), max(indexes)//5))
    # ax1.scatter(indexes, test_accs, marker='o', s=0.5, c='tab:blue')
    # ax1.set_xlabel('architecture ID', fontsize=LabelSize)
    # ax1.set_ylabel('test accuracy (%)', fontsize=LabelSize)

    ax2.scatter(params, train_accs, marker="o", s=0.5, c="tab:blue")
    ax2.scatter(
        [params[x] for x in resnet_indexes],
        [train_accs[x] for x in resnet_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="ResNet",
        alpha=xalpha,
    )
    ax2.scatter(
        [params[x] for x in largest_indexes],
        [train_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax2.set_xlabel("#parameters (MB)", fontsize=LabelSize)
    ax2.set_ylabel("train accuracy (%)", fontsize=LabelSize)
    ax2.legend(loc=4, fontsize=LegendFontsize)

    ax3.scatter(params, test_accs, marker="o", s=0.5, c="tab:blue")
    ax3.scatter(
        [params[x] for x in resnet_indexes],
        [test_accs[x] for x in resnet_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="ResNet",
        alpha=xalpha,
    )
    ax3.scatter(
        [params[x] for x in largest_indexes],
        [test_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax3.set_xlabel("#parameters (MB)", fontsize=LabelSize)
    ax3.set_ylabel("test accuracy (%)", fontsize=LabelSize)
    ax3.legend(loc=4, fontsize=LegendFontsize)

    ax4.scatter(flops, train_accs, marker="o", s=0.5, c="tab:blue")
    ax4.scatter(
        [flops[x] for x in resnet_indexes],
        [train_accs[x] for x in resnet_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="ResNet",
        alpha=xalpha,
    )
    ax4.scatter(
        [flops[x] for x in largest_indexes],
        [train_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax4.set_xlabel("#FLOPs (M)", fontsize=LabelSize)
    ax4.set_ylabel("train accuracy (%)", fontsize=LabelSize)
    ax4.legend(loc=4, fontsize=LegendFontsize)

    ax5.scatter(flops, test_accs, marker="o", s=0.5, c="tab:blue")
    ax5.scatter(
        [flops[x] for x in resnet_indexes],
        [test_accs[x] for x in resnet_indexes],
        marker="*",
        s=xscale,
        c="tab:orange",
        label="ResNet",
        alpha=xalpha,
    )
    ax5.scatter(
        [flops[x] for x in largest_indexes],
        [test_accs[x] for x in largest_indexes],
        marker="x",
        s=xscale,
        c="tab:green",
        label="Largest Candidate",
        alpha=xalpha,
    )
    ax5.set_xlabel("#FLOPs (M)", fontsize=LabelSize)
    ax5.set_ylabel("test accuracy (%)", fontsize=LabelSize)
    ax5.legend(loc=4, fontsize=LegendFontsize)

    save_path = vis_save_dir / "tss-{:}.png".format(dataset)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


def visualize_rank_info(api, vis_save_dir, indicator):
    vis_save_dir = vis_save_dir.resolve()
    # print ('{:} start to visualize {:} information'.format(time_string(), api))
    vis_save_dir.mkdir(parents=True, exist_ok=True)

    cifar010_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "cifar10", indicator
    )
    cifar100_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "cifar100", indicator
    )
    imagenet_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "ImageNet16-120", indicator
    )
    cifar010_info = torch.load(cifar010_cache_path)
    cifar100_info = torch.load(cifar100_cache_path)
    imagenet_info = torch.load(imagenet_cache_path)
    indexes = list(range(len(cifar010_info["params"])))

    print("{:} start to visualize relative ranking".format(time_string()))

    dpi, width, height = 250, 3800, 1200
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 14, 14

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    ax1, ax2, ax3 = axs

    def get_labels(info):
        ord_test_indexes = sorted(indexes, key=lambda i: info["test_accs"][i])
        ord_valid_indexes = sorted(indexes, key=lambda i: info["valid_accs"][i])
        labels = []
        for idx in ord_test_indexes:
            labels.append(ord_valid_indexes.index(idx))
        return labels

    def plot_ax(labels, ax, name):
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize)
            tick.label.set_rotation(90)
        ax.set_xlim(min(indexes), max(indexes))
        ax.set_ylim(min(indexes), max(indexes))
        ax.yaxis.set_ticks(np.arange(min(indexes), max(indexes), max(indexes) // 3))
        ax.xaxis.set_ticks(np.arange(min(indexes), max(indexes), max(indexes) // 5))
        ax.scatter(indexes, labels, marker="^", s=0.5, c="tab:green", alpha=0.8)
        ax.scatter(indexes, indexes, marker="o", s=0.5, c="tab:blue", alpha=0.8)
        ax.scatter(
            [-1], [-1], marker="^", s=100, c="tab:green", label="{:} test".format(name)
        )
        ax.scatter(
            [-1],
            [-1],
            marker="o",
            s=100,
            c="tab:blue",
            label="{:} validation".format(name),
        )
        ax.legend(loc=4, fontsize=LegendFontsize)
        ax.set_xlabel("ranking on the {:} validation".format(name), fontsize=LabelSize)
        ax.set_ylabel("architecture ranking", fontsize=LabelSize)

    labels = get_labels(cifar010_info)
    plot_ax(labels, ax1, "CIFAR-10")
    labels = get_labels(cifar100_info)
    plot_ax(labels, ax2, "CIFAR-100")
    labels = get_labels(imagenet_info)
    plot_ax(labels, ax3, "ImageNet-16-120")

    save_path = (
        vis_save_dir / "{:}-same-relative-rank.pdf".format(indicator)
    ).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (
        vis_save_dir / "{:}-same-relative-rank.png".format(indicator)
    ).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


def calculate_correlation(*vectors):
    matrix = []
    for i, vectori in enumerate(vectors):
        x = []
        for j, vectorj in enumerate(vectors):
            x.append(np.corrcoef(vectori, vectorj)[0, 1])
        matrix.append(x)
    return np.array(matrix)


def visualize_all_rank_info(api, vis_save_dir, indicator):
    vis_save_dir = vis_save_dir.resolve()
    # print ('{:} start to visualize {:} information'.format(time_string(), api))
    vis_save_dir.mkdir(parents=True, exist_ok=True)

    cifar010_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "cifar10", indicator
    )
    cifar100_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "cifar100", indicator
    )
    imagenet_cache_path = vis_save_dir / "{:}-cache-{:}-info.pth".format(
        "ImageNet16-120", indicator
    )
    cifar010_info = torch.load(cifar010_cache_path)
    cifar100_info = torch.load(cifar100_cache_path)
    imagenet_info = torch.load(imagenet_cache_path)
    indexes = list(range(len(cifar010_info["params"])))

    print("{:} start to visualize relative ranking".format(time_string()))

    dpi, width, height = 250, 3200, 1400
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 14, 14

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = axs

    sns_size = 15
    CoRelMatrix = calculate_correlation(
        cifar010_info["valid_accs"],
        cifar010_info["test_accs"],
        cifar100_info["valid_accs"],
        cifar100_info["test_accs"],
        imagenet_info["valid_accs"],
        imagenet_info["test_accs"],
    )

    sns.heatmap(
        CoRelMatrix,
        annot=True,
        annot_kws={"size": sns_size},
        fmt=".3f",
        linewidths=0.5,
        ax=ax1,
        xticklabels=["C10-V", "C10-T", "C100-V", "C100-T", "I120-V", "I120-T"],
        yticklabels=["C10-V", "C10-T", "C100-V", "C100-T", "I120-V", "I120-T"],
    )

    selected_indexes, acc_bar = [], 92
    for i, acc in enumerate(cifar010_info["test_accs"]):
        if acc > acc_bar:
            selected_indexes.append(i)
    cifar010_valid_accs = np.array(cifar010_info["valid_accs"])[selected_indexes]
    cifar010_test_accs = np.array(cifar010_info["test_accs"])[selected_indexes]
    cifar100_valid_accs = np.array(cifar100_info["valid_accs"])[selected_indexes]
    cifar100_test_accs = np.array(cifar100_info["test_accs"])[selected_indexes]
    imagenet_valid_accs = np.array(imagenet_info["valid_accs"])[selected_indexes]
    imagenet_test_accs = np.array(imagenet_info["test_accs"])[selected_indexes]
    CoRelMatrix = calculate_correlation(
        cifar010_valid_accs,
        cifar010_test_accs,
        cifar100_valid_accs,
        cifar100_test_accs,
        imagenet_valid_accs,
        imagenet_test_accs,
    )

    sns.heatmap(
        CoRelMatrix,
        annot=True,
        annot_kws={"size": sns_size},
        fmt=".3f",
        linewidths=0.5,
        ax=ax2,
        xticklabels=["C10-V", "C10-T", "C100-V", "C100-T", "I120-V", "I120-T"],
        yticklabels=["C10-V", "C10-T", "C100-V", "C100-T", "I120-V", "I120-T"],
    )
    ax1.set_title("Correlation coefficient over ALL candidates")
    ax2.set_title(
        "Correlation coefficient over candidates with accuracy > {:}%".format(acc_bar)
    )
    save_path = (vis_save_dir / "{:}-all-relative-rank.png".format(indicator)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NAS-Bench-X",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/vis-nas-bench",
        help="Folder to save checkpoints and log.",
    )
    # use for train the model
    args = parser.parse_args()

    to_save_dir = Path(args.save_dir)

    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    api201 = create(None, "tss", verbose=True)
    for xdata in datasets:
        visualize_tss_info(api201, xdata, to_save_dir)

    api_sss = create(None, "size", verbose=True)
    for xdata in datasets:
        visualize_sss_info(api_sss, xdata, to_save_dir)

    visualize_info(None, to_save_dir, "tss")
    visualize_info(None, to_save_dir, "sss")
    visualize_rank_info(None, to_save_dir, "tss")
    visualize_rank_info(None, to_save_dir, "sss")

    visualize_all_rank_info(None, to_save_dir, "tss")
    visualize_all_rank_info(None, to_save_dir, "sss")
