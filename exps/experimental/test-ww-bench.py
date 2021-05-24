#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
###########################################################################################################################################################
# Before run these commands, the files must be properly put.
#
# CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python exps/experimental/test-ww-bench.py --search_space sss --base_path $HOME/.torch/NATS-tss-v1_0-3ffb9 --dataset cifar10
# CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python exps/experimental/test-ww-bench.py --search_space sss --base_path $HOME/.torch/NATS-sss-v1_0-50262 --dataset cifar100
# CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python exps/experimental/test-ww-bench.py --search_space sss --base_path $HOME/.torch/NATS-sss-v1_0-50262 --dataset ImageNet16-120
# CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python exps/experimental/test-ww-bench.py --search_space tss --base_path $HOME/.torch/NATS-tss-v1_0-3ffb9 --dataset cifar10
###########################################################################################################################################################
import os, gc, sys, math, argparse, psutil
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict
import matplotlib
import seaborn as sns

matplotlib.use("agg")
import matplotlib.pyplot as plt

lib_dir = (Path(__file__).parent / ".." / "..").resolve()
print("LIB-DIR: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from log_utils import time_string
from nats_bench import create
from models import get_cell_based_tiny_net
from utils import weight_watcher


"""
def get_cor(A, B):
  return float(np.corrcoef(A, B)[0,1])


def tostr(accdict, norms):
  xstr = []
  for key, accs in accdict.items():
    cor = get_cor(accs, norms)
    xstr.append('{:}: {:.3f}'.format(key, cor))
  return ' '.join(xstr)
"""


def evaluate(api, weight_dir, data: str):
    print("\nEvaluate dataset={:}".format(data))
    process = psutil.Process(os.getpid())
    norms, accuracies = [], []
    ok, total = 0, 5000
    for idx in range(total):
        arch_index = api.random()
        api.reload(weight_dir, arch_index)
        # compute the weight watcher results
        config = api.get_net_config(arch_index, data)
        net = get_cell_based_tiny_net(config)
        meta_info = api.query_meta_info_by_index(
            arch_index, hp="200" if api.search_space_name == "topology" else "90"
        )
        params = meta_info.get_net_param(
            data, 888 if api.search_space_name == "topology" else 777
        )
        with torch.no_grad():
            net.load_state_dict(params)
            _, summary = weight_watcher.analyze(net, alphas=False)
            if "lognorm" not in summary:
                api.clear_params(arch_index, None)
                del net
                continue
                continue
            cur_norm = -summary["lognorm"]
        api.clear_params(arch_index, None)
        if math.isnan(cur_norm):
            del net, meta_info
            continue
        else:
            ok += 1
            norms.append(cur_norm)
        # query the accuracy
        info = meta_info.get_metrics(
            data,
            "ori-test",
            iepoch=None,
            is_random=888 if api.search_space_name == "topology" else 777,
        )
        accuracies.append(info["accuracy"])
        del net, meta_info
        # print the information
        if idx % 20 == 0:
            gc.collect()
            print(
                "{:} {:04d}_{:04d}/{:04d} ({:.2f} MB memory)".format(
                    time_string(), ok, idx, total, process.memory_info().rss / 1e6
                )
            )
    return norms, accuracies


def main(search_space, meta_file: str, weight_dir, save_dir, xdata):
    save_dir.mkdir(parents=True, exist_ok=True)
    api = create(meta_file, search_space, verbose=False)
    datasets = ["cifar10-valid", "cifar10", "cifar100", "ImageNet16-120"]
    print(time_string() + " " + "=" * 50)
    for data in datasets:
        hps = api.avaliable_hps
        for hp in hps:
            nums = api.statistics(data, hp=hp)
            total = sum([k * v for k, v in nums.items()])
            print(
                "Using {:3s} epochs, trained on {:20s} : {:} trials in total ({:}).".format(
                    hp, data, total, nums
                )
            )
    print(time_string() + " " + "=" * 50)

    norms, accuracies = evaluate(api, weight_dir, xdata)

    indexes = list(range(len(norms)))
    norm_indexes = sorted(indexes, key=lambda i: norms[i])
    accy_indexes = sorted(indexes, key=lambda i: accuracies[i])
    labels = []
    for index in norm_indexes:
        labels.append(accy_indexes.index(index))

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
    ax.scatter(indexes, labels, marker="*", s=0.5, c="tab:red", alpha=0.8)
    ax.scatter(indexes, indexes, marker="o", s=0.5, c="tab:blue", alpha=0.8)
    ax.scatter([-1], [-1], marker="o", s=100, c="tab:blue", label="Test accuracy")
    ax.scatter([-1], [-1], marker="*", s=100, c="tab:red", label="Weight watcher")
    plt.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.legend(loc=0, fontsize=LegendFontsize)
    ax.set_xlabel(
        "architecture ranking sorted by the test accuracy ", fontsize=LabelSize
    )
    ax.set_ylabel("architecture ranking computed by weight watcher", fontsize=LabelSize)
    save_path = (save_dir / "{:}-{:}-test-ww.pdf".format(search_space, xdata)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="pdf")
    save_path = (save_dir / "{:}-{:}-test-ww.png".format(search_space, xdata)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
    print("{:} save into {:}".format(time_string(), save_path))

    print("{:} finish this test.".format(time_string()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis of NAS-Bench-201")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/vis-nas-bench/",
        help="The base-name of folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        default=None,
        choices=["tss", "sss"],
        help="The search space.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="The path to the NAS-Bench-201 benchmark file and weight dir.",
    )
    parser.add_argument("--dataset", type=str, default=None, help=".")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_file = Path(args.base_path + ".pth")
    weight_dir = Path(args.base_path + "-full")
    assert meta_file.exists(), "invalid path for api : {:}".format(meta_file)
    assert (
        weight_dir.exists() and weight_dir.is_dir()
    ), "invalid path for weight dir : {:}".format(weight_dir)

    main(args.search_space, str(meta_file), weight_dir, save_dir, args.dataset)
