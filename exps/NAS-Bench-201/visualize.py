#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
# python exps/NAS-Bench-201/visualize.py --api_path $HOME/.torch/NAS-Bench-201-v1_0-e61699.pth
#####################################################
import sys, argparse
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use("agg")
import matplotlib.pyplot as plt

from xautodl.log_utils import time_string
from nas_201_api import NASBench201API as API


def calculate_correlation(*vectors):
    matrix = []
    for i, vectori in enumerate(vectors):
        x = []
        for j, vectorj in enumerate(vectors):
            x.append(np.corrcoef(vectori, vectorj)[0, 1])
        matrix.append(x)
    return np.array(matrix)


def visualize_relative_ranking(vis_save_dir):
    print("\n" + "-" * 100)
    cifar010_cache_path = vis_save_dir / "{:}-cache-info.pth".format("cifar10")
    cifar100_cache_path = vis_save_dir / "{:}-cache-info.pth".format("cifar100")
    imagenet_cache_path = vis_save_dir / "{:}-cache-info.pth".format("ImageNet16-120")
    cifar010_info = torch.load(cifar010_cache_path)
    cifar100_info = torch.load(cifar100_cache_path)
    imagenet_info = torch.load(imagenet_cache_path)
    indexes = list(range(len(cifar010_info["params"])))

    print("{:} start to visualize relative ranking".format(time_string()))
    # maximum accuracy with ResNet-level params 11472
    x_010_accs = [
        cifar010_info["test_accs"][i]
        if cifar010_info["params"][i] <= cifar010_info["params"][11472]
        else -1
        for i in indexes
    ]
    x_100_accs = [
        cifar100_info["test_accs"][i]
        if cifar100_info["params"][i] <= cifar100_info["params"][11472]
        else -1
        for i in indexes
    ]
    x_img_accs = [
        imagenet_info["test_accs"][i]
        if imagenet_info["params"][i] <= imagenet_info["params"][11472]
        else -1
        for i in indexes
    ]

    cifar010_ord_indexes = sorted(indexes, key=lambda i: cifar010_info["test_accs"][i])
    cifar100_ord_indexes = sorted(indexes, key=lambda i: cifar100_info["test_accs"][i])
    imagenet_ord_indexes = sorted(indexes, key=lambda i: imagenet_info["test_accs"][i])

    cifar100_labels, imagenet_labels = [], []
    for idx in cifar010_ord_indexes:
        cifar100_labels.append(cifar100_ord_indexes.index(idx))
        imagenet_labels.append(imagenet_ord_indexes.index(idx))
    print("{:} prepare data done.".format(time_string()))

    dpi, width, height = 300, 2600, 2600
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 18, 18
    resnet_scale, resnet_alpha = 120, 0.5

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xlim(min(indexes), max(indexes))
    plt.ylim(min(indexes), max(indexes))
    # plt.ylabel('y').set_rotation(0)
    plt.yticks(
        np.arange(min(indexes), max(indexes), max(indexes) // 6),
        fontsize=LegendFontsize,
        rotation="vertical",
    )
    plt.xticks(
        np.arange(min(indexes), max(indexes), max(indexes) // 6),
        fontsize=LegendFontsize,
    )
    # ax.scatter(indexes, cifar100_labels, marker='^', s=0.5, c='tab:green', alpha=0.8, label='CIFAR-100')
    # ax.scatter(indexes, imagenet_labels, marker='*', s=0.5, c='tab:red'  , alpha=0.8, label='ImageNet-16-120')
    # ax.scatter(indexes, indexes        , marker='o', s=0.5, c='tab:blue' , alpha=0.8, label='CIFAR-10')
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
    save_path = (vis_save_dir / "relative-rank.pdf").resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (vis_save_dir / "relative-rank.png").resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))

    # calculate correlation
    sns_size = 15
    CoRelMatrix = calculate_correlation(
        cifar010_info["valid_accs"],
        cifar010_info["test_accs"],
        cifar100_info["valid_accs"],
        cifar100_info["test_accs"],
        imagenet_info["valid_accs"],
        imagenet_info["test_accs"],
    )
    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    h = sns.heatmap(
        CoRelMatrix, annot=True, annot_kws={"size": sns_size}, fmt=".3f", linewidths=0.5
    )
    save_path = (vis_save_dir / "co-relation-all.pdf").resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    print("{:} save into {:}".format(time_string(), save_path))

    # calculate correlation
    acc_bars = [92, 93]
    for acc_bar in acc_bars:
        selected_indexes = []
        for i, acc in enumerate(cifar010_info["test_accs"]):
            if acc > acc_bar:
                selected_indexes.append(i)
        print("select {:} architectures".format(len(selected_indexes)))
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
        fig = plt.figure(figsize=figsize)
        plt.axis("off")
        h = sns.heatmap(
            CoRelMatrix,
            annot=True,
            annot_kws={"size": sns_size},
            fmt=".3f",
            linewidths=0.5,
        )
        save_path = (
            vis_save_dir / "co-relation-top-{:}.pdf".format(len(selected_indexes))
        ).resolve()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
        print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


def visualize_info(meta_file, dataset, vis_save_dir):
    print("{:} start to visualize {:} information".format(time_string(), dataset))
    cache_file_path = vis_save_dir / "{:}-cache-info.pth".format(dataset)
    if not cache_file_path.exists():
        print("Do not find cache file : {:}".format(cache_file_path))
        nas_bench = API(str(meta_file))
        params, flops, train_accs, valid_accs, test_accs, otest_accs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for index in range(len(nas_bench)):
            info = nas_bench.query_by_index(index, use_12epochs_result=False)
            resx = info.get_comput_costs(dataset)
            flop, param = resx["flops"], resx["params"]
            if dataset == "cifar10":
                res = info.get_metrics("cifar10", "train")
                train_acc = res["accuracy"]
                res = info.get_metrics("cifar10-valid", "x-valid")
                valid_acc = res["accuracy"]
                res = info.get_metrics("cifar10", "ori-test")
                test_acc = res["accuracy"]
                res = info.get_metrics("cifar10", "ori-test")
                otest_acc = res["accuracy"]
            else:
                res = info.get_metrics(dataset, "train")
                train_acc = res["accuracy"]
                res = info.get_metrics(dataset, "x-valid")
                valid_acc = res["accuracy"]
                res = info.get_metrics(dataset, "x-test")
                test_acc = res["accuracy"]
                res = info.get_metrics(dataset, "ori-test")
                otest_acc = res["accuracy"]
            if index == 11472:  # resnet
                resnet = {
                    "params": param,
                    "flops": flop,
                    "index": 11472,
                    "train_acc": train_acc,
                    "valid_acc": valid_acc,
                    "test_acc": test_acc,
                    "otest_acc": otest_acc,
                }
            flops.append(flop)
            params.append(param)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            test_accs.append(test_acc)
            otest_accs.append(otest_acc)
        # resnet = {'params': 0.559, 'flops': 78.56, 'index': 11472, 'train_acc': 99.99, 'valid_acc': 90.84, 'test_acc': 93.97}
        info = {
            "params": params,
            "flops": flops,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "test_accs": test_accs,
            "otest_accs": otest_accs,
        }
        info["resnet"] = resnet
        torch.save(info, cache_file_path)
    else:
        print("Find cache file : {:}".format(cache_file_path))
        info = torch.load(cache_file_path)
        params, flops, train_accs, valid_accs, test_accs, otest_accs = (
            info["params"],
            info["flops"],
            info["train_accs"],
            info["valid_accs"],
            info["test_accs"],
            info["otest_accs"],
        )
        resnet = info["resnet"]
    print("{:} collect data done.".format(time_string()))

    indexes = list(range(len(params)))
    dpi, width, height = 300, 2600, 2600
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 22, 22
    resnet_scale, resnet_alpha = 120, 0.5

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(0, 1.6, 0.3), fontsize=LegendFontsize)
    if dataset == "cifar10":
        plt.ylim(50, 100)
        plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
    elif dataset == "cifar100":
        plt.ylim(25, 75)
        plt.yticks(np.arange(25, 76, 10), fontsize=LegendFontsize)
    else:
        plt.ylim(0, 50)
        plt.yticks(np.arange(0, 51, 10), fontsize=LegendFontsize)
    ax.scatter(params, valid_accs, marker="o", s=0.5, c="tab:blue")
    ax.scatter(
        [resnet["params"]],
        [resnet["valid_acc"]],
        marker="*",
        s=resnet_scale,
        c="tab:orange",
        label="resnet",
        alpha=0.4,
    )
    plt.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.legend(loc=4, fontsize=LegendFontsize)
    ax.set_xlabel("#parameters (MB)", fontsize=LabelSize)
    ax.set_ylabel("the validation accuracy (%)", fontsize=LabelSize)
    save_path = (vis_save_dir / "{:}-param-vs-valid.pdf".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (vis_save_dir / "{:}-param-vs-valid.png".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(0, 1.6, 0.3), fontsize=LegendFontsize)
    if dataset == "cifar10":
        plt.ylim(50, 100)
        plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
    elif dataset == "cifar100":
        plt.ylim(25, 75)
        plt.yticks(np.arange(25, 76, 10), fontsize=LegendFontsize)
    else:
        plt.ylim(0, 50)
        plt.yticks(np.arange(0, 51, 10), fontsize=LegendFontsize)
    ax.scatter(params, test_accs, marker="o", s=0.5, c="tab:blue")
    ax.scatter(
        [resnet["params"]],
        [resnet["test_acc"]],
        marker="*",
        s=resnet_scale,
        c="tab:orange",
        label="resnet",
        alpha=resnet_alpha,
    )
    plt.grid()
    ax.set_axisbelow(True)
    plt.legend(loc=4, fontsize=LegendFontsize)
    ax.set_xlabel("#parameters (MB)", fontsize=LabelSize)
    ax.set_ylabel("the test accuracy (%)", fontsize=LabelSize)
    save_path = (vis_save_dir / "{:}-param-vs-test.pdf".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (vis_save_dir / "{:}-param-vs-test.png".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(0, 1.6, 0.3), fontsize=LegendFontsize)
    if dataset == "cifar10":
        plt.ylim(50, 100)
        plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
    elif dataset == "cifar100":
        plt.ylim(20, 100)
        plt.yticks(np.arange(20, 101, 10), fontsize=LegendFontsize)
    else:
        plt.ylim(25, 76)
        plt.yticks(np.arange(25, 76, 10), fontsize=LegendFontsize)
    ax.scatter(params, train_accs, marker="o", s=0.5, c="tab:blue")
    ax.scatter(
        [resnet["params"]],
        [resnet["train_acc"]],
        marker="*",
        s=resnet_scale,
        c="tab:orange",
        label="resnet",
        alpha=resnet_alpha,
    )
    plt.grid()
    ax.set_axisbelow(True)
    plt.legend(loc=4, fontsize=LegendFontsize)
    ax.set_xlabel("#parameters (MB)", fontsize=LabelSize)
    ax.set_ylabel("the trarining accuracy (%)", fontsize=LabelSize)
    save_path = (vis_save_dir / "{:}-param-vs-train.pdf".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (vis_save_dir / "{:}-param-vs-train.png".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xlim(0, max(indexes))
    plt.xticks(
        np.arange(min(indexes), max(indexes), max(indexes) // 5),
        fontsize=LegendFontsize,
    )
    if dataset == "cifar10":
        plt.ylim(50, 100)
        plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
    elif dataset == "cifar100":
        plt.ylim(25, 75)
        plt.yticks(np.arange(25, 76, 10), fontsize=LegendFontsize)
    else:
        plt.ylim(0, 50)
        plt.yticks(np.arange(0, 51, 10), fontsize=LegendFontsize)
    ax.scatter(indexes, test_accs, marker="o", s=0.5, c="tab:blue")
    ax.scatter(
        [resnet["index"]],
        [resnet["test_acc"]],
        marker="*",
        s=resnet_scale,
        c="tab:orange",
        label="resnet",
        alpha=resnet_alpha,
    )
    plt.grid()
    ax.set_axisbelow(True)
    plt.legend(loc=4, fontsize=LegendFontsize)
    ax.set_xlabel("architecture ID", fontsize=LabelSize)
    ax.set_ylabel("the test accuracy (%)", fontsize=LabelSize)
    save_path = (vis_save_dir / "{:}-test-over-ID.pdf".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (vis_save_dir / "{:}-test-over-ID.png".format(dataset)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))
    plt.close("all")


def visualize_rank_over_time(meta_file, vis_save_dir):
    print("\n" + "-" * 150)
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    print(
        "{:} start to visualize rank-over-time into {:}".format(
            time_string(), vis_save_dir
        )
    )
    cache_file_path = vis_save_dir / "rank-over-time-cache-info.pth"
    if not cache_file_path.exists():
        print("Do not find cache file : {:}".format(cache_file_path))
        nas_bench = API(str(meta_file))
        print("{:} load nas_bench done".format(time_string()))
        params, flops, train_accs, valid_accs, test_accs, otest_accs = (
            [],
            [],
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        # for iepoch in range(200): for index in range( len(nas_bench) ):
        for index in tqdm(range(len(nas_bench))):
            info = nas_bench.query_by_index(index, use_12epochs_result=False)
            for iepoch in range(200):
                res = info.get_metrics("cifar10", "train", iepoch)
                train_acc = res["accuracy"]
                res = info.get_metrics("cifar10-valid", "x-valid", iepoch)
                valid_acc = res["accuracy"]
                res = info.get_metrics("cifar10", "ori-test", iepoch)
                test_acc = res["accuracy"]
                res = info.get_metrics("cifar10", "ori-test", iepoch)
                otest_acc = res["accuracy"]
                train_accs[iepoch].append(train_acc)
                valid_accs[iepoch].append(valid_acc)
                test_accs[iepoch].append(test_acc)
                otest_accs[iepoch].append(otest_acc)
                if iepoch == 0:
                    res = info.get_comput_costs("cifar10")
                    flop, param = res["flops"], res["params"]
                    flops.append(flop)
                    params.append(param)
        info = {
            "params": params,
            "flops": flops,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "test_accs": test_accs,
            "otest_accs": otest_accs,
        }
        torch.save(info, cache_file_path)
    else:
        print("Find cache file : {:}".format(cache_file_path))
        info = torch.load(cache_file_path)
        params, flops, train_accs, valid_accs, test_accs, otest_accs = (
            info["params"],
            info["flops"],
            info["train_accs"],
            info["valid_accs"],
            info["test_accs"],
            info["otest_accs"],
        )
    print("{:} collect data done.".format(time_string()))
    # selected_epochs = [0, 100, 150, 180, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
    selected_epochs = list(range(200))
    x_xtests = test_accs[199]
    indexes = list(range(len(x_xtests)))
    ord_idxs = sorted(indexes, key=lambda i: x_xtests[i])
    for sepoch in selected_epochs:
        x_valids = valid_accs[sepoch]
        valid_ord_idxs = sorted(indexes, key=lambda i: x_valids[i])
        valid_ord_lbls = []
        for idx in ord_idxs:
            valid_ord_lbls.append(valid_ord_idxs.index(idx))
        # labeled data
        dpi, width, height = 300, 2600, 2600
        figsize = width / float(dpi), height / float(dpi)
        LabelSize, LegendFontsize = 18, 18

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.xlim(min(indexes), max(indexes))
        plt.ylim(min(indexes), max(indexes))
        plt.yticks(
            np.arange(min(indexes), max(indexes), max(indexes) // 6),
            fontsize=LegendFontsize,
            rotation="vertical",
        )
        plt.xticks(
            np.arange(min(indexes), max(indexes), max(indexes) // 6),
            fontsize=LegendFontsize,
        )
        ax.scatter(indexes, valid_ord_lbls, marker="^", s=0.5, c="tab:green", alpha=0.8)
        ax.scatter(indexes, indexes, marker="o", s=0.5, c="tab:blue", alpha=0.8)
        ax.scatter(
            [-1], [-1], marker="^", s=100, c="tab:green", label="CIFAR-10 validation"
        )
        ax.scatter([-1], [-1], marker="o", s=100, c="tab:blue", label="CIFAR-10 test")
        plt.grid(zorder=0)
        ax.set_axisbelow(True)
        plt.legend(loc="upper left", fontsize=LegendFontsize)
        ax.set_xlabel(
            "architecture ranking in the final test accuracy", fontsize=LabelSize
        )
        ax.set_ylabel("architecture ranking in the validation set", fontsize=LabelSize)
        save_path = (vis_save_dir / "time-{:03d}.pdf".format(sepoch)).resolve()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
        save_path = (vis_save_dir / "time-{:03d}.png".format(sepoch)).resolve()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
        print("{:} save into {:}".format(time_string(), save_path))
        plt.close("all")


def write_video(save_dir):
    import cv2

    video_save_path = save_dir / "time.avi"
    print("{:} start create video for {:}".format(time_string(), video_save_path))
    images = sorted(list(save_dir.glob("time-*.png")))
    ximage = cv2.imread(str(images[0]))
    # shape  = (ximage.shape[1], ximage.shape[0])
    shape = (1000, 1000)
    # writer = cv2.VideoWriter(str(video_save_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, shape)
    writer = cv2.VideoWriter(
        str(video_save_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, shape
    )
    for idx, image in enumerate(images):
        ximage = cv2.imread(str(image))
        _image = cv2.resize(ximage, shape)
        writer.write(_image)
    writer.release()
    print("write video [{:} frames] into {:}".format(len(images), video_save_path))


def plot_results_nas_v2(api, dataset_xset_a, dataset_xset_b, root, file_name, y_lims):
    # print ('root-path={:}, dataset={:}, xset={:}'.format(root, dataset, xset))
    print("root-path : {:} and {:}".format(dataset_xset_a, dataset_xset_b))
    checkpoints = [
        "./output/search-cell-nas-bench-201/R-EA-cifar10/results.pth",
        "./output/search-cell-nas-bench-201/REINFORCE-cifar10/results.pth",
        "./output/search-cell-nas-bench-201/RAND-cifar10/results.pth",
        "./output/search-cell-nas-bench-201/BOHB-cifar10/results.pth",
    ]
    legends, indexes = ["REA", "REINFORCE", "RANDOM", "BOHB"], None
    All_Accs_A, All_Accs_B = OrderedDict(), OrderedDict()
    for legend, checkpoint in zip(legends, checkpoints):
        all_indexes = torch.load(checkpoint, map_location="cpu")
        accuracies_A, accuracies_B = [], []
        accuracies = []
        for x in all_indexes:
            info = api.arch2infos_full[x]
            metrics = info.get_metrics(
                dataset_xset_a[0], dataset_xset_a[1], None, False
            )
            accuracies_A.append(metrics["accuracy"])
            metrics = info.get_metrics(
                dataset_xset_b[0], dataset_xset_b[1], None, False
            )
            accuracies_B.append(metrics["accuracy"])
            accuracies.append((accuracies_A[-1], accuracies_B[-1]))
        if indexes is None:
            indexes = list(range(len(all_indexes)))
        accuracies = sorted(accuracies)
        All_Accs_A[legend] = [x[0] for x in accuracies]
        All_Accs_B[legend] = [x[1] for x in accuracies]

    color_set = ["r", "b", "g", "c", "m", "y", "k"]
    dpi, width, height = 300, 3400, 2600
    LabelSize, LegendFontsize = 28, 28
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    x_axis = np.arange(0, 600)
    plt.xlim(0, max(indexes))
    plt.ylim(y_lims[0], y_lims[1])
    interval_x, interval_y = 100, y_lims[2]
    plt.xticks(np.arange(0, max(indexes), interval_x), fontsize=LegendFontsize)
    plt.yticks(np.arange(y_lims[0], y_lims[1], interval_y), fontsize=LegendFontsize)
    plt.grid()
    plt.xlabel("The index of runs", fontsize=LabelSize)
    plt.ylabel("The accuracy (%)", fontsize=LabelSize)

    for idx, legend in enumerate(legends):
        plt.plot(
            indexes,
            All_Accs_B[legend],
            color=color_set[idx],
            linestyle="--",
            label="{:}".format(legend),
            lw=1,
            alpha=0.5,
        )
        plt.plot(indexes, All_Accs_A[legend], color=color_set[idx], linestyle="-", lw=1)
        for All_Accs in [All_Accs_A, All_Accs_B]:
            print(
                "{:} : mean = {:}, std = {:} :: {:.2f}$\\pm${:.2f}".format(
                    legend,
                    np.mean(All_Accs[legend]),
                    np.std(All_Accs[legend]),
                    np.mean(All_Accs[legend]),
                    np.std(All_Accs[legend]),
                )
            )
    plt.legend(loc=4, fontsize=LegendFontsize)
    save_path = root / "{:}".format(file_name)
    print("save figure into {:}\n".format(save_path))
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", format="pdf")


def plot_results_nas(api, dataset, xset, root, file_name, y_lims):
    print("root-path={:}, dataset={:}, xset={:}".format(root, dataset, xset))
    checkpoints = [
        "./output/search-cell-nas-bench-201/R-EA-cifar10/results.pth",
        "./output/search-cell-nas-bench-201/REINFORCE-cifar10/results.pth",
        "./output/search-cell-nas-bench-201/RAND-cifar10/results.pth",
        "./output/search-cell-nas-bench-201/BOHB-cifar10/results.pth",
    ]
    legends, indexes = ["REA", "REINFORCE", "RANDOM", "BOHB"], None
    All_Accs = OrderedDict()
    for legend, checkpoint in zip(legends, checkpoints):
        all_indexes = torch.load(checkpoint, map_location="cpu")
        accuracies = []
        for x in all_indexes:
            info = api.arch2infos_full[x]
            metrics = info.get_metrics(dataset, xset, None, False)
            accuracies.append(metrics["accuracy"])
        if indexes is None:
            indexes = list(range(len(all_indexes)))
        All_Accs[legend] = sorted(accuracies)

    color_set = ["r", "b", "g", "c", "m", "y", "k"]
    dpi, width, height = 300, 3400, 2600
    LabelSize, LegendFontsize = 28, 28
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    x_axis = np.arange(0, 600)
    plt.xlim(0, max(indexes))
    plt.ylim(y_lims[0], y_lims[1])
    interval_x, interval_y = 100, y_lims[2]
    plt.xticks(np.arange(0, max(indexes), interval_x), fontsize=LegendFontsize)
    plt.yticks(np.arange(y_lims[0], y_lims[1], interval_y), fontsize=LegendFontsize)
    plt.grid()
    plt.xlabel("The index of runs", fontsize=LabelSize)
    plt.ylabel("The accuracy (%)", fontsize=LabelSize)

    for idx, legend in enumerate(legends):
        plt.plot(
            indexes,
            All_Accs[legend],
            color=color_set[idx],
            linestyle="-",
            label="{:}".format(legend),
            lw=2,
        )
        print(
            "{:} : mean = {:}, std = {:} :: {:.2f}$\\pm${:.2f}".format(
                legend,
                np.mean(All_Accs[legend]),
                np.std(All_Accs[legend]),
                np.mean(All_Accs[legend]),
                np.std(All_Accs[legend]),
            )
        )
    plt.legend(loc=4, fontsize=LegendFontsize)
    save_path = root / "{:}-{:}-{:}".format(dataset, xset, file_name)
    print("save figure into {:}\n".format(save_path))
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", format="pdf")


def just_show(api):
    xtimes = {
        "RSPS": [8082.5, 7794.2, 8144.7],
        "DARTS-V1": [11582.1, 11347.0, 11948.2],
        "DARTS-V2": [35694.7, 36132.7, 35518.0],
        "GDAS": [31334.1, 31478.6, 32016.7],
        "SETN": [33528.8, 33831.5, 35058.3],
        "ENAS": [14340.2, 13817.3, 14018.9],
    }
    for xkey, xlist in xtimes.items():
        xlist = np.array(xlist)
        print("{:4s} : mean-time={:.2f} s".format(xkey, xlist.mean()))

    xpaths = {
        "RSPS": "output/search-cell-nas-bench-201/RANDOM-NAS-cifar10/checkpoint/",
        "DARTS-V1": "output/search-cell-nas-bench-201/DARTS-V1-cifar10/checkpoint/",
        "DARTS-V2": "output/search-cell-nas-bench-201/DARTS-V2-cifar10/checkpoint/",
        "GDAS": "output/search-cell-nas-bench-201/GDAS-cifar10/checkpoint/",
        "SETN": "output/search-cell-nas-bench-201/SETN-cifar10/checkpoint/",
        "ENAS": "output/search-cell-nas-bench-201/ENAS-cifar10/checkpoint/",
    }
    xseeds = {
        "RSPS": [5349, 59613, 5983],
        "DARTS-V1": [11416, 72873, 81184],
        "DARTS-V2": [43330, 79405, 79423],
        "GDAS": [19677, 884, 95950],
        "SETN": [20518, 61817, 89144],
        "ENAS": [3231, 34238, 96929],
    }

    def get_accs(xdata, index=-1):
        if index == -1:
            epochs = xdata["epoch"]
            genotype = xdata["genotypes"][epochs - 1]
            index = api.query_index_by_arch(genotype)
        pairs = [
            ("cifar10-valid", "x-valid"),
            ("cifar10", "ori-test"),
            ("cifar100", "x-valid"),
            ("cifar100", "x-test"),
            ("ImageNet16-120", "x-valid"),
            ("ImageNet16-120", "x-test"),
        ]
        xresults = []
        for dataset, xset in pairs:
            metrics = api.arch2infos_full[index].get_metrics(dataset, xset, None, False)
            xresults.append(metrics["accuracy"])
        return xresults

    for xkey in xpaths.keys():
        all_paths = [
            "{:}/seed-{:}-basic.pth".format(xpaths[xkey], seed) for seed in xseeds[xkey]
        ]
        all_datas = [torch.load(xpath) for xpath in all_paths]
        accyss = [get_accs(xdatas) for xdatas in all_datas]
        accyss = np.array(accyss)
        print("\nxkey = {:}".format(xkey))
        for i in range(accyss.shape[1]):
            print(
                "---->>>> {:.2f}$\\pm${:.2f}".format(
                    accyss[:, i].mean(), accyss[:, i].std()
                )
            )

    print("\n{:}".format(get_accs(None, 11472)))  # resnet
    pairs = [
        ("cifar10-valid", "x-valid"),
        ("cifar10", "ori-test"),
        ("cifar100", "x-valid"),
        ("cifar100", "x-test"),
        ("ImageNet16-120", "x-valid"),
        ("ImageNet16-120", "x-test"),
    ]
    for dataset, metric_on_set in pairs:
        arch_index, highest_acc = api.find_best(dataset, metric_on_set)
        print(
            "[{:10s}-{:10s} ::: index={:5d}, accuracy={:.2f}".format(
                dataset, metric_on_set, arch_index, highest_acc
            )
        )


def show_nas_sharing_w(
    api, dataset, subset, vis_save_dir, sufix, file_name, y_lims, x_maxs
):
    color_set = ["r", "b", "g", "c", "m", "y", "k"]
    dpi, width, height = 300, 3400, 2600
    LabelSize, LegendFontsize = 28, 28
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    # x_maxs = 250
    plt.xlim(0, x_maxs + 1)
    plt.ylim(y_lims[0], y_lims[1])
    interval_x, interval_y = x_maxs // 5, y_lims[2]
    plt.xticks(np.arange(0, x_maxs + 1, interval_x), fontsize=LegendFontsize)
    plt.yticks(np.arange(y_lims[0], y_lims[1], interval_y), fontsize=LegendFontsize)
    plt.grid()
    plt.xlabel("The searching epoch", fontsize=LabelSize)
    plt.ylabel("The accuracy (%)", fontsize=LabelSize)

    xpaths = {
        "RSPS": "output/search-cell-nas-bench-201/RANDOM-NAS-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "DARTS-V1": "output/search-cell-nas-bench-201/DARTS-V1-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "DARTS-V2": "output/search-cell-nas-bench-201/DARTS-V2-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "GDAS": "output/search-cell-nas-bench-201/GDAS-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "SETN": "output/search-cell-nas-bench-201/SETN-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "ENAS": "output/search-cell-nas-bench-201/ENAS-cifar10-{:}/checkpoint/".format(
            sufix
        ),
    }
    """
  xseeds = {'RSPS'    : [5349, 59613, 5983],
            'DARTS-V1': [11416, 72873, 81184, 28640],
            'DARTS-V2': [43330, 79405, 79423],
            'GDAS'    : [19677, 884, 95950],
            'SETN'    : [20518, 61817, 89144],
            'ENAS'    : [3231, 34238, 96929],
           }
  """
    xseeds = {
        "RSPS": [23814, 28015, 95809],
        "DARTS-V1": [48349, 80877, 81920],
        "DARTS-V2": [61712, 7941, 87041],
        "GDAS": [72818, 72996, 78877],
        "SETN": [26985, 55206, 95404],
        "ENAS": [21792, 36605, 45029],
    }

    def get_accs(xdata):
        epochs, xresults = xdata["epoch"], []
        if -1 in xdata["genotypes"]:
            metrics = api.arch2infos_full[
                api.query_index_by_arch(xdata["genotypes"][-1])
            ].get_metrics(dataset, subset, None, False)
        else:
            metrics = api.arch2infos_full[api.random()].get_metrics(
                dataset, subset, None, False
            )
        xresults.append(metrics["accuracy"])
        for iepoch in range(epochs):
            genotype = xdata["genotypes"][iepoch]
            index = api.query_index_by_arch(genotype)
            metrics = api.arch2infos_full[index].get_metrics(
                dataset, subset, None, False
            )
            xresults.append(metrics["accuracy"])
        return xresults

    if x_maxs == 50:
        xox, xxxstrs = "v2", ["DARTS-V1", "DARTS-V2"]
    elif x_maxs == 250:
        xox, xxxstrs = "v1", ["RSPS", "GDAS", "SETN", "ENAS"]
    else:
        raise ValueError("invalid x_maxs={:}".format(x_maxs))

    for idx, method in enumerate(xxxstrs):
        xkey = method
        all_paths = [
            "{:}/seed-{:}-basic.pth".format(xpaths[xkey], seed) for seed in xseeds[xkey]
        ]
        all_datas = [torch.load(xpath, map_location="cpu") for xpath in all_paths]
        accyss = [get_accs(xdatas) for xdatas in all_datas]
        accyss = np.array(accyss)
        epochs = list(range(accyss.shape[1]))
        plt.plot(
            epochs,
            [accyss[:, i].mean() for i in epochs],
            color=color_set[idx],
            linestyle="-",
            label="{:}".format(method),
            lw=2,
        )
        plt.fill_between(
            epochs,
            [accyss[:, i].mean() - accyss[:, i].std() for i in epochs],
            [accyss[:, i].mean() + accyss[:, i].std() for i in epochs],
            alpha=0.2,
            color=color_set[idx],
        )
    # plt.legend(loc=4, fontsize=LegendFontsize)
    plt.legend(loc=0, fontsize=LegendFontsize)
    save_path = vis_save_dir / "{:}.pdf".format(file_name)
    print("save figure into {:}\n".format(save_path))
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", format="pdf")


def show_nas_sharing_w_v2(
    api, data_sub_a, data_sub_b, vis_save_dir, sufix, file_name, y_lims, x_maxs
):
    color_set = ["r", "b", "g", "c", "m", "y", "k"]
    dpi, width, height = 300, 3400, 2600
    LabelSize, LegendFontsize = 28, 28
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    # x_maxs = 250
    plt.xlim(0, x_maxs + 1)
    plt.ylim(y_lims[0], y_lims[1])
    interval_x, interval_y = x_maxs // 5, y_lims[2]
    plt.xticks(np.arange(0, x_maxs + 1, interval_x), fontsize=LegendFontsize)
    plt.yticks(np.arange(y_lims[0], y_lims[1], interval_y), fontsize=LegendFontsize)
    plt.grid()
    plt.xlabel("The searching epoch", fontsize=LabelSize)
    plt.ylabel("The accuracy (%)", fontsize=LabelSize)

    xpaths = {
        "RSPS": "output/search-cell-nas-bench-201/RANDOM-NAS-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "DARTS-V1": "output/search-cell-nas-bench-201/DARTS-V1-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "DARTS-V2": "output/search-cell-nas-bench-201/DARTS-V2-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "GDAS": "output/search-cell-nas-bench-201/GDAS-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "SETN": "output/search-cell-nas-bench-201/SETN-cifar10-{:}/checkpoint/".format(
            sufix
        ),
        "ENAS": "output/search-cell-nas-bench-201/ENAS-cifar10-{:}/checkpoint/".format(
            sufix
        ),
    }
    """
  xseeds = {'RSPS'    : [5349, 59613, 5983],
            'DARTS-V1': [11416, 72873, 81184, 28640],
            'DARTS-V2': [43330, 79405, 79423],
            'GDAS'    : [19677, 884, 95950],
            'SETN'    : [20518, 61817, 89144],
            'ENAS'    : [3231, 34238, 96929],
           }
  """
    xseeds = {
        "RSPS": [23814, 28015, 95809],
        "DARTS-V1": [48349, 80877, 81920],
        "DARTS-V2": [61712, 7941, 87041],
        "GDAS": [72818, 72996, 78877],
        "SETN": [26985, 55206, 95404],
        "ENAS": [21792, 36605, 45029],
    }

    def get_accs(xdata, dataset, subset):
        epochs, xresults = xdata["epoch"], []
        if -1 in xdata["genotypes"]:
            metrics = api.arch2infos_full[
                api.query_index_by_arch(xdata["genotypes"][-1])
            ].get_metrics(dataset, subset, None, False)
        else:
            metrics = api.arch2infos_full[api.random()].get_metrics(
                dataset, subset, None, False
            )
        xresults.append(metrics["accuracy"])
        for iepoch in range(epochs):
            genotype = xdata["genotypes"][iepoch]
            index = api.query_index_by_arch(genotype)
            metrics = api.arch2infos_full[index].get_metrics(
                dataset, subset, None, False
            )
            xresults.append(metrics["accuracy"])
        return xresults

    if x_maxs == 50:
        xox, xxxstrs = "v2", ["DARTS-V1", "DARTS-V2"]
    elif x_maxs == 250:
        xox, xxxstrs = "v1", ["RSPS", "GDAS", "SETN", "ENAS"]
    else:
        raise ValueError("invalid x_maxs={:}".format(x_maxs))

    for idx, method in enumerate(xxxstrs):
        xkey = method
        all_paths = [
            "{:}/seed-{:}-basic.pth".format(xpaths[xkey], seed) for seed in xseeds[xkey]
        ]
        all_datas = [torch.load(xpath, map_location="cpu") for xpath in all_paths]
        accyss_A = np.array(
            [get_accs(xdatas, data_sub_a[0], data_sub_a[1]) for xdatas in all_datas]
        )
        accyss_B = np.array(
            [get_accs(xdatas, data_sub_b[0], data_sub_b[1]) for xdatas in all_datas]
        )
        epochs = list(range(accyss_A.shape[1]))
        for j, accyss in enumerate([accyss_A, accyss_B]):
            if x_maxs == 50:
                color, line = color_set[idx * 2 + j], "-" if j == 0 else "--"
            elif x_maxs == 250:
                color, line = color_set[idx], "-" if j == 0 else "--"
            else:
                raise ValueError("invalid x-maxs={:}".format(x_maxs))
            plt.plot(
                epochs,
                [accyss[:, i].mean() for i in epochs],
                color=color,
                linestyle=line,
                label="{:} ({:})".format(method, "VALID" if j == 0 else "TEST"),
                lw=2,
                alpha=0.9,
            )
            plt.fill_between(
                epochs,
                [accyss[:, i].mean() - accyss[:, i].std() for i in epochs],
                [accyss[:, i].mean() + accyss[:, i].std() for i in epochs],
                alpha=0.2,
                color=color,
            )
            setname = data_sub_a if j == 0 else data_sub_b
            print(
                "{:} -- {:} ---- {:.2f}$\\pm${:.2f}".format(
                    method, setname, accyss[:, -1].mean(), accyss[:, -1].std()
                )
            )
    # plt.legend(loc=4, fontsize=LegendFontsize)
    plt.legend(loc=0, fontsize=LegendFontsize)
    save_path = vis_save_dir / "{:}-{:}".format(xox, file_name)
    print("save figure into {:}\n".format(save_path))
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", format="pdf")


def show_reinforce(api, root, dataset, xset, file_name, y_lims):
    print("root-path={:}, dataset={:}, xset={:}".format(root, dataset, xset))
    LRs = ["0.01", "0.02", "0.1", "0.2", "0.5"]
    checkpoints = [
        "./output/search-cell-nas-bench-201/REINFORCE-cifar10-{:}/results.pth".format(x)
        for x in LRs
    ]
    acc_lr_dict, indexes = {}, None
    for lr, checkpoint in zip(LRs, checkpoints):
        all_indexes, accuracies = torch.load(checkpoint, map_location="cpu"), []
        for x in all_indexes:
            info = api.arch2infos_full[x]
            metrics = info.get_metrics(dataset, xset, None, False)
            accuracies.append(metrics["accuracy"])
        if indexes is None:
            indexes = list(range(len(accuracies)))
        acc_lr_dict[lr] = np.array(sorted(accuracies))
        print(
            "LR={:.3f}, mean={:}, std={:}".format(
                float(lr), acc_lr_dict[lr].mean(), acc_lr_dict[lr].std()
            )
        )

    color_set = ["r", "b", "g", "c", "m", "y", "k"]
    dpi, width, height = 300, 3400, 2600
    LabelSize, LegendFontsize = 28, 22
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    x_axis = np.arange(0, 600)
    plt.xlim(0, max(indexes))
    plt.ylim(y_lims[0], y_lims[1])
    interval_x, interval_y = 100, y_lims[2]
    plt.xticks(np.arange(0, max(indexes), interval_x), fontsize=LegendFontsize)
    plt.yticks(np.arange(y_lims[0], y_lims[1], interval_y), fontsize=LegendFontsize)
    plt.grid()
    plt.xlabel("The index of runs", fontsize=LabelSize)
    plt.ylabel("The accuracy (%)", fontsize=LabelSize)

    for idx, LR in enumerate(LRs):
        legend = "LR={:.2f}".format(float(LR))
        # color, linestyle = color_set[idx // 2], '-' if idx % 2 == 0 else '-.'
        color, linestyle = color_set[idx], "-"
        plt.plot(
            indexes,
            acc_lr_dict[LR],
            color=color,
            linestyle=linestyle,
            label=legend,
            lw=2,
            alpha=0.8,
        )
        print(
            "{:} : mean = {:}, std = {:} :: {:.2f}$\\pm${:.2f}".format(
                legend,
                np.mean(acc_lr_dict[LR]),
                np.std(acc_lr_dict[LR]),
                np.mean(acc_lr_dict[LR]),
                np.std(acc_lr_dict[LR]),
            )
        )
    plt.legend(loc=4, fontsize=LegendFontsize)
    save_path = root / "{:}-{:}-{:}.pdf".format(dataset, xset, file_name)
    print("save figure into {:}\n".format(save_path))
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", format="pdf")


def show_rea(api, root, dataset, xset, file_name, y_lims):
    print("root-path={:}, dataset={:}, xset={:}".format(root, dataset, xset))
    SSs = [3, 5, 10]
    checkpoints = [
        "./output/search-cell-nas-bench-201/R-EA-cifar10-SS{:}/results.pth".format(x)
        for x in SSs
    ]
    acc_ss_dict, indexes = {}, None
    for ss, checkpoint in zip(SSs, checkpoints):
        all_indexes, accuracies = torch.load(checkpoint, map_location="cpu"), []
        for x in all_indexes:
            info = api.arch2infos_full[x]
            metrics = info.get_metrics(dataset, xset, None, False)
            accuracies.append(metrics["accuracy"])
        if indexes is None:
            indexes = list(range(len(accuracies)))
        acc_ss_dict[ss] = np.array(sorted(accuracies))
        print(
            "Sample-Size={:2d}, mean={:}, std={:}".format(
                ss, acc_ss_dict[ss].mean(), acc_ss_dict[ss].std()
            )
        )

    color_set = ["r", "b", "g", "c", "m", "y", "k"]
    dpi, width, height = 300, 3400, 2600
    LabelSize, LegendFontsize = 28, 22
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    x_axis = np.arange(0, 600)
    plt.xlim(0, max(indexes))
    plt.ylim(y_lims[0], y_lims[1])
    interval_x, interval_y = 100, y_lims[2]
    plt.xticks(np.arange(0, max(indexes), interval_x), fontsize=LegendFontsize)
    plt.yticks(np.arange(y_lims[0], y_lims[1], interval_y), fontsize=LegendFontsize)
    plt.grid()
    plt.xlabel("The index of runs", fontsize=LabelSize)
    plt.ylabel("The accuracy (%)", fontsize=LabelSize)

    for idx, ss in enumerate(SSs):
        legend = "sample-size={:2d}".format(ss)
        # color, linestyle = color_set[idx // 2], '-' if idx % 2 == 0 else '-.'
        color, linestyle = color_set[idx], "-"
        plt.plot(
            indexes,
            acc_ss_dict[ss],
            color=color,
            linestyle=linestyle,
            label=legend,
            lw=2,
            alpha=0.8,
        )
        print(
            "{:} : mean = {:}, std = {:} :: {:.2f}$\\pm${:.2f}".format(
                legend,
                np.mean(acc_ss_dict[ss]),
                np.std(acc_ss_dict[ss]),
                np.mean(acc_ss_dict[ss]),
                np.std(acc_ss_dict[ss]),
            )
        )
    plt.legend(loc=4, fontsize=LegendFontsize)
    save_path = root / "{:}-{:}-{:}.pdf".format(dataset, xset, file_name)
    print("save figure into {:}\n".format(save_path))
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", format="pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NAS-Bench-201",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search-cell-nas-bench-201/visuals",
        help="The base-name of folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--api_path",
        type=str,
        default=None,
        help="The path to the NAS-Bench-201 benchmark file.",
    )
    args = parser.parse_args()

    vis_save_dir = Path(args.save_dir)
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    meta_file = Path(args.api_path)
    assert meta_file.exists(), "invalid path for api : {:}".format(meta_file)
    # visualize_rank_over_time(str(meta_file), vis_save_dir / 'over-time')
    # write_video(vis_save_dir / 'over-time')
    # visualize_info(str(meta_file), 'cifar10' , vis_save_dir)
    # visualize_info(str(meta_file), 'cifar100', vis_save_dir)
    # visualize_info(str(meta_file), 'ImageNet16-120', vis_save_dir)
    # visualize_relative_ranking(vis_save_dir)

    api = API(args.api_path)
    # show_reinforce(api, vis_save_dir, 'cifar10-valid' , 'x-valid', 'REINFORCE-CIFAR-10', (85, 92, 2))
    # show_rea      (api, vis_save_dir, 'cifar10-valid' , 'x-valid', 'REA-CIFAR-10', (88, 92, 1))

    # plot_results_nas_v2(api, ('cifar10-valid' , 'x-valid'), ('cifar10'       , 'ori-test'), vis_save_dir, 'nas-com-v2-cifar010.pdf', (85,95, 1))
    # plot_results_nas_v2(api, ('cifar100'      , 'x-valid'), ('cifar100'      , 'x-test'  ), vis_save_dir, 'nas-com-v2-cifar100.pdf', (60,75, 3))
    # plot_results_nas_v2(api, ('ImageNet16-120', 'x-valid'), ('ImageNet16-120', 'x-test'  ), vis_save_dir, 'nas-com-v2-imagenet.pdf', (35,48, 2))

    show_nas_sharing_w_v2(
        api,
        ("cifar10-valid", "x-valid"),
        ("cifar10", "ori-test"),
        vis_save_dir,
        "BN0",
        "BN0-DARTS-CIFAR010.pdf",
        (0, 100, 10),
        50,
    )
    show_nas_sharing_w_v2(
        api,
        ("cifar100", "x-valid"),
        ("cifar100", "x-test"),
        vis_save_dir,
        "BN0",
        "BN0-DARTS-CIFAR100.pdf",
        (0, 100, 10),
        50,
    )
    show_nas_sharing_w_v2(
        api,
        ("ImageNet16-120", "x-valid"),
        ("ImageNet16-120", "x-test"),
        vis_save_dir,
        "BN0",
        "BN0-DARTS-ImageNet.pdf",
        (0, 100, 10),
        50,
    )

    show_nas_sharing_w_v2(
        api,
        ("cifar10-valid", "x-valid"),
        ("cifar10", "ori-test"),
        vis_save_dir,
        "BN0",
        "BN0-OTHER-CIFAR010.pdf",
        (0, 100, 10),
        250,
    )
    show_nas_sharing_w_v2(
        api,
        ("cifar100", "x-valid"),
        ("cifar100", "x-test"),
        vis_save_dir,
        "BN0",
        "BN0-OTHER-CIFAR100.pdf",
        (0, 100, 10),
        250,
    )
    show_nas_sharing_w_v2(
        api,
        ("ImageNet16-120", "x-valid"),
        ("ImageNet16-120", "x-test"),
        vis_save_dir,
        "BN0",
        "BN0-OTHER-ImageNet.pdf",
        (0, 100, 10),
        250,
    )

    show_nas_sharing_w(
        api,
        "cifar10-valid",
        "x-valid",
        vis_save_dir,
        "BN0",
        "BN0-XX-CIFAR010-VALID.pdf",
        (0, 100, 10),
        250,
    )
    show_nas_sharing_w(
        api,
        "cifar10",
        "ori-test",
        vis_save_dir,
        "BN0",
        "BN0-XX-CIFAR010-TEST.pdf",
        (0, 100, 10),
        250,
    )
    """
  for x_maxs in [50, 250]:
    show_nas_sharing_w(api, 'cifar10-valid' , 'x-valid' , vis_save_dir, 'nas-plot.pdf', (0, 100,10), x_maxs)
    show_nas_sharing_w(api, 'cifar10'       , 'ori-test', vis_save_dir, 'nas-plot.pdf', (0, 100,10), x_maxs)
    show_nas_sharing_w(api, 'cifar100'      , 'x-valid' , vis_save_dir, 'nas-plot.pdf', (0, 100,10), x_maxs)
    show_nas_sharing_w(api, 'cifar100'      , 'x-test'  , vis_save_dir, 'nas-plot.pdf', (0, 100,10), x_maxs)
    show_nas_sharing_w(api, 'ImageNet16-120', 'x-valid' , vis_save_dir, 'nas-plot.pdf', (0, 100,10), x_maxs)
    show_nas_sharing_w(api, 'ImageNet16-120', 'x-test'  , vis_save_dir, 'nas-plot.pdf', (0, 100,10), x_maxs)
  
  show_nas_sharing_w_v2(api, ('cifar10-valid' , 'x-valid'), ('cifar10'       , 'ori-test') , vis_save_dir, 'DARTS-CIFAR010.pdf', (0, 100,10), 50)
  just_show(api)
  plot_results_nas(api, 'cifar10-valid' , 'x-valid' , vis_save_dir, 'nas-com.pdf', (85,95, 1))
  plot_results_nas(api, 'cifar10'       , 'ori-test', vis_save_dir, 'nas-com.pdf', (85,95, 1))
  plot_results_nas(api, 'cifar100'      , 'x-valid' , vis_save_dir, 'nas-com.pdf', (55,75, 3))
  plot_results_nas(api, 'cifar100'      , 'x-test'  , vis_save_dir, 'nas-com.pdf', (55,75, 3))
  plot_results_nas(api, 'ImageNet16-120', 'x-valid' , vis_save_dir, 'nas-com.pdf', (35,50, 3))
  plot_results_nas(api, 'ImageNet16-120', 'x-test'  , vis_save_dir, 'nas-com.pdf', (35,50, 3))
  """
