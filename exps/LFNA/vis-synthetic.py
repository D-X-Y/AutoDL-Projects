#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
############################################################################
# python exps/LFNA/vis-synthetic.py                                        #
############################################################################
import os, sys, copy, random
import torch
import numpy as np
import argparse
from collections import OrderedDict, defaultdict
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

import matplotlib
from matplotlib import cm

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


from datasets.synthetic_core import get_synthetic_env
from datasets.synthetic_example import create_example_v1
from utils.temp_sync import optimize_fn, evaluate_fn
from procedures.metric_utils import MSEMetric


def plot_scatter(cur_ax, xs, ys, color, alpha, linewidths, label=None):
    cur_ax.scatter([-100], [-100], color=color, linewidths=linewidths, label=label)
    cur_ax.scatter(xs, ys, color=color, alpha=alpha, linewidths=1.5, label=None)


def draw_multi_fig(save_dir, timestamp, scatter_list, wh, fig_title=None):
    save_path = save_dir / "{:04d}".format(timestamp)
    # print('Plot the figure at timestamp-{:} into {:}'.format(timestamp, save_path))
    dpi, width, height = 40, wh[0], wh[1]
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize, font_gap = 80, 80, 5

    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(
            fig_title, fontsize=LegendFontsize, fontweight="bold", x=0.5, y=0.92
        )

    for idx, scatter_dict in enumerate(scatter_list):
        cur_ax = fig.add_subplot(len(scatter_list), 1, idx + 1)
        plot_scatter(
            cur_ax,
            scatter_dict["xaxis"],
            scatter_dict["yaxis"],
            scatter_dict["color"],
            scatter_dict["alpha"],
            scatter_dict["linewidths"],
            scatter_dict["label"],
        )
        cur_ax.set_xlabel("X", fontsize=LabelSize)
        cur_ax.set_ylabel("Y", rotation=0, fontsize=LabelSize)
        cur_ax.set_xlim(scatter_dict["xlim"][0], scatter_dict["xlim"][1])
        cur_ax.set_ylim(scatter_dict["ylim"][0], scatter_dict["ylim"][1])
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
            tick.label.set_rotation(10)
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
        cur_ax.legend(loc=1, fontsize=LegendFontsize)
    fig.savefig(str(save_path) + ".pdf", dpi=dpi, bbox_inches="tight", format="pdf")
    fig.savefig(str(save_path) + ".png", dpi=dpi, bbox_inches="tight", format="png")
    plt.close("all")


def find_min(cur, others):
    if cur is None:
        return float(others)
    else:
        return float(min(cur, others))


def find_max(cur, others):
    if cur is None:
        return float(others.max())
    else:
        return float(max(cur, others))


def compare_cl(save_dir):
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    dynamic_env, cl_function = create_example_v1(
        # timestamp_config=dict(num=200, min_timestamp=-1, max_timestamp=1.0),
        timestamp_config=dict(num=200),
        num_per_task=1000,
    )

    models = dict()

    cl_function.set_timestamp(0)
    cl_xaxis_min = None
    cl_xaxis_max = None

    all_data = OrderedDict()

    for idx, (timestamp, dataset) in enumerate(tqdm(dynamic_env, ncols=50)):
        xaxis_all = dataset[0][:, 0].numpy()
        yaxis_all = dataset[1][:, 0].numpy()
        current_data = dict()
        current_data["lfna_xaxis_all"] = xaxis_all
        current_data["lfna_yaxis_all"] = yaxis_all

        # compute cl-min
        cl_xaxis_min = find_min(cl_xaxis_min, xaxis_all.mean() - xaxis_all.std())
        cl_xaxis_max = find_max(cl_xaxis_max, xaxis_all.mean() + xaxis_all.std())
        all_data[timestamp] = current_data

    global_cl_xaxis_all = np.arange(cl_xaxis_min, cl_xaxis_max, step=0.1)
    global_cl_yaxis_all = cl_function.noise_call(global_cl_xaxis_all)

    for idx, (timestamp, xdata) in enumerate(tqdm(all_data.items(), ncols=50)):
        scatter_list = []
        scatter_list.append(
            {
                "xaxis": xdata["lfna_xaxis_all"],
                "yaxis": xdata["lfna_yaxis_all"],
                "color": "k",
                "linewidths": 15,
                "alpha": 0.99,
                "xlim": (-6, 6),
                "ylim": (-40, 40),
                "label": "LFNA",
            }
        )

        cur_cl_xaxis_min = cl_xaxis_min
        cur_cl_xaxis_max = cl_xaxis_min + (cl_xaxis_max - cl_xaxis_min) * (
            idx + 1
        ) / len(all_data)
        cl_xaxis_all = np.arange(cur_cl_xaxis_min, cur_cl_xaxis_max, step=0.01)
        cl_yaxis_all = cl_function.noise_call(cl_xaxis_all, std=0.2)

        scatter_list.append(
            {
                "xaxis": cl_xaxis_all,
                "yaxis": cl_yaxis_all,
                "color": "k",
                "linewidths": 15,
                "xlim": (round(cl_xaxis_min, 1), round(cl_xaxis_max, 1)),
                "ylim": (-20, 6),
                "alpha": 0.99,
                "label": "Continual Learning",
            }
        )

        draw_multi_fig(
            save_dir,
            idx,
            scatter_list,
            wh=(2200, 1800),
            fig_title="Timestamp={:03d}".format(idx),
        )
    print("Save all figures into {:}".format(save_dir))
    save_dir = save_dir.resolve()
    base_cmd = (
        "ffmpeg -y -i {xdir}/%04d.png -vf fps=1 -vf scale=2200:1800 -vb 5000k".format(
            xdir=save_dir
        )
    )
    video_cmd = "{:} -pix_fmt yuv420p {xdir}/compare-cl.mp4".format(
        base_cmd, xdir=save_dir
    )
    print(video_cmd + "\n")
    os.system(video_cmd)
    os.system(
        "{:} -pix_fmt yuv420p {xdir}/compare-cl.webm".format(base_cmd, xdir=save_dir)
    )


def visualize_env(save_dir):
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    dynamic_env = get_synthetic_env()
    min_t, max_t = dynamic_env.min_timestamp, dynamic_env.max_timestamp
    for idx, (timestamp, (allx, ally)) in enumerate(tqdm(dynamic_env, ncols=50)):
        dpi, width, height = 30, 1800, 1400
        figsize = width / float(dpi), height / float(dpi)
        LabelSize, LegendFontsize, font_gap = 80, 80, 5
        fig = plt.figure(figsize=figsize)

        cur_ax = fig.add_subplot(1, 1, 1)
        allx, ally = allx[:, 0].numpy(), ally[:, 0].numpy()
        plot_scatter(cur_ax, allx, ally, "k", 0.99, 15, "timestamp={:05d}".format(idx))
        cur_ax.set_xlabel("X", fontsize=LabelSize)
        cur_ax.set_ylabel("Y", rotation=0, fontsize=LabelSize)
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
            tick.label.set_rotation(10)
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
        cur_ax.set_xlim(-10, 10)
        cur_ax.set_ylim(-60, 60)
        cur_ax.legend(loc=1, fontsize=LegendFontsize)

        save_path = save_dir / "{:05d}".format(idx)
        fig.savefig(str(save_path) + ".pdf", dpi=dpi, bbox_inches="tight", format="pdf")
        fig.savefig(str(save_path) + ".png", dpi=dpi, bbox_inches="tight", format="png")
        plt.close("all")
    save_dir = save_dir.resolve()
    base_cmd = "ffmpeg -y -i {xdir}/%05d.png -vf scale=1800:1400 -pix_fmt yuv420p -vb 5000k".format(
        xdir=save_dir
    )
    os.system("{:} {xdir}/env.mp4".format(base_cmd, xdir=save_dir))
    os.system("{:} {xdir}/env.webm".format(base_cmd, xdir=save_dir))


def compare_algs(save_dir, alg_dir="./outputs/lfna-synthetic"):
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    dpi, width, height = 30, 1800, 1400
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize, font_gap = 80, 80, 5

    cache_path = Path(alg_dir) / "env-info.pth"
    assert cache_path.exists(), "{:} does not exist".format(cache_path)
    env_info = torch.load(cache_path)

    alg_name2dir = OrderedDict()
    alg_name2dir["Optimal"] = "use-same-timestamp"
    alg_name2dir["History SL"] = "use-all-past-data"
    colors = ["r", "g"]

    dynamic_env = env_info["dynamic_env"]
    min_t, max_t = dynamic_env.min_timestamp, dynamic_env.max_timestamp

    linewidths = 10
    for idx, (timestamp, (ori_allx, ori_ally)) in enumerate(
        tqdm(dynamic_env, ncols=50)
    ):
        if idx == 0:
            continue
        fig = plt.figure(figsize=figsize)
        cur_ax = fig.add_subplot(1, 1, 1)

        # the data
        allx, ally = ori_allx[:, 0].numpy(), ori_ally[:, 0].numpy()
        plot_scatter(cur_ax, allx, ally, "k", 0.99, linewidths, "Raw Data")

        for idx_alg, (alg, xdir) in enumerate(alg_name2dir.items()):
            ckp_path = (
                Path(alg_dir)
                / xdir
                / "{:04d}-{:04d}.pth".format(idx, env_info["total"])
            )
            assert ckp_path.exists()
            ckp_data = torch.load(ckp_path)
            with torch.no_grad():
                predicts = ckp_data["model"](ori_allx)
                predicts = predicts.cpu().view(-1).numpy()
            plot_scatter(cur_ax, allx, predicts, colors[idx_alg], 0.99, linewidths, alg)

        cur_ax.set_xlabel("X", fontsize=LabelSize)
        cur_ax.set_ylabel("Y", rotation=0, fontsize=LabelSize)
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
            tick.label.set_rotation(10)
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
        cur_ax.set_xlim(-10, 10)
        cur_ax.set_ylim(-60, 60)
        cur_ax.legend(loc=1, fontsize=LegendFontsize)

        save_path = save_dir / "{:05d}".format(idx)
        fig.savefig(str(save_path) + ".pdf", dpi=dpi, bbox_inches="tight", format="pdf")
        fig.savefig(str(save_path) + ".png", dpi=dpi, bbox_inches="tight", format="png")
        plt.close("all")
    save_dir = save_dir.resolve()
    base_cmd = "ffmpeg -y -i {xdir}/%05d.png -vf scale={w}:{h} -pix_fmt yuv420p -vb 5000k".format(
        xdir=save_dir, w=width, h=height
    )
    os.system("{:} {xdir}/compare-alg.mp4".format(base_cmd, xdir=save_dir))
    os.system("{:} {xdir}/compare-alg.webm".format(base_cmd, xdir=save_dir))


def compare_algs_v2(save_dir, alg_dir="./outputs/lfna-synthetic"):
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    dpi, width, height = 30, 3200, 2000
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize, font_gap = 80, 80, 5

    cache_path = Path(alg_dir) / "env-info.pth"
    assert cache_path.exists(), "{:} does not exist".format(cache_path)
    env_info = torch.load(cache_path)

    alg_name2dir = OrderedDict()
    alg_name2dir["Optimal"] = "use-same-timestamp"
    alg_name2dir["Supervised Learning (History Data)"] = "use-all-past-data"
    colors = ["r", "g"]

    alg2xs, alg2ys = defaultdict(list), defaultdict(list)
    colors = ["r", "g"]

    dynamic_env = env_info["dynamic_env"]
    min_t, max_t = dynamic_env.min_timestamp, dynamic_env.max_timestamp

    linewidths = 10
    for idx, (timestamp, (ori_allx, ori_ally)) in enumerate(
        tqdm(dynamic_env, ncols=50)
    ):
        if idx == 0:
            continue
        fig = plt.figure(figsize=figsize)
        cur_ax = fig.add_subplot(2, 1, 1)

        # the data
        allx, ally = ori_allx[:, 0].numpy(), ori_ally[:, 0].numpy()
        plot_scatter(cur_ax, allx, ally, "k", 0.99, linewidths, "Raw Data")

        for idx_alg, (alg, xdir) in enumerate(alg_name2dir.items()):
            ckp_path = (
                Path(alg_dir)
                / xdir
                / "{:04d}-{:04d}.pth".format(idx, env_info["total"])
            )
            assert ckp_path.exists()
            ckp_data = torch.load(ckp_path)
            with torch.no_grad():
                predicts = ckp_data["model"](ori_allx)
                predicts = predicts.cpu()
                # keep data
                metric = MSEMetric()
                metric(predicts, ori_ally)
                predicts = predicts.view(-1).numpy()
                alg2xs[alg].append(idx)
                alg2ys[alg].append(metric.get_info()["mse"])
            plot_scatter(cur_ax, allx, predicts, colors[idx_alg], 0.99, linewidths, alg)

        cur_ax.set_xlabel("X", fontsize=LabelSize)
        cur_ax.set_ylabel("Y", rotation=0, fontsize=LabelSize)
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
            tick.label.set_rotation(10)
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
        cur_ax.set_xlim(-10, 10)
        cur_ax.set_ylim(-60, 60)
        cur_ax.legend(loc=1, fontsize=LegendFontsize)

        # the trajectory data
        cur_ax = fig.add_subplot(2, 1, 2)
        for idx_alg, (alg, xdir) in enumerate(alg_name2dir.items()):
            # plot_scatter(cur_ax, alg2xs[alg], alg2ys[alg], olors[idx_alg], 0.99, linewidths, alg)
            cur_ax.plot(
                alg2xs[alg],
                alg2ys[alg],
                color=colors[idx_alg],
                linestyle="-",
                linewidth=5,
                label=alg,
            )
        cur_ax.legend(loc=1, fontsize=LegendFontsize)

        cur_ax.set_xlabel("Timestamp", fontsize=LabelSize)
        cur_ax.set_ylabel("MSE", fontsize=LabelSize)
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
            tick.label.set_rotation(10)
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
        cur_ax.set_xlim(1, len(dynamic_env))
        cur_ax.set_ylim(0, 10)
        cur_ax.legend(loc=1, fontsize=LegendFontsize)

        save_path = save_dir / "{:05d}".format(idx)
        fig.savefig(str(save_path) + ".pdf", dpi=dpi, bbox_inches="tight", format="pdf")
        fig.savefig(str(save_path) + ".png", dpi=dpi, bbox_inches="tight", format="png")
        plt.close("all")
    save_dir = save_dir.resolve()
    base_cmd = "ffmpeg -y -i {xdir}/%05d.png -vf scale={w}:{h} -pix_fmt yuv420p -vb 5000k".format(
        xdir=save_dir, w=width, h=height
    )
    os.system("{:} {xdir}/compare-alg.mp4".format(base_cmd, xdir=save_dir))
    os.system("{:} {xdir}/compare-alg.webm".format(base_cmd, xdir=save_dir))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visualize synthetic data.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/vis-synthetic",
        help="The save directory.",
    )
    args = parser.parse_args()

    compare_algs_v2(os.path.join(args.save_dir, "compare-alg-v2"))
    # visualize_env(os.path.join(args.save_dir, "vis-env"))
    # compare_cl(os.path.join(args.save_dir, "compare-cl"))
    # compare_algs(os.path.join(args.save_dir, "compare-alg"))
