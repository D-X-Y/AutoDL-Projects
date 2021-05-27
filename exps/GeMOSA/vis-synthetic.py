#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
############################################################################
# python exps/GeMOSA/vis-synthetic.py --env_version v1                     #
# python exps/GeMOSA/vis-synthetic.py --env_version v2                     #
# python exps/GeMOSA/vis-synthetic.py --env_version v3                     #
# python exps/GeMOSA/vis-synthetic.py --env_version v4                     #
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

lib_dir = (Path(__file__).parent / ".." / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from xautodl.models.xcore import get_model
from xautodl.datasets.synthetic_core import get_synthetic_env
from xautodl.procedures.metric_utils import MSEMetric


def plot_scatter(cur_ax, xs, ys, color, alpha, linewidths, label=None):
    cur_ax.scatter([-100], [-100], color=color, linewidths=linewidths[0], label=label)
    cur_ax.scatter(
        xs, ys, color=color, alpha=alpha, linewidths=linewidths[1], label=None
    )


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


def visualize_env(save_dir, version):
    save_dir = Path(str(save_dir))
    for substr in ("pdf", "png"):
        sub_save_dir = save_dir / "{:}-{:}".format(substr, version)
        sub_save_dir.mkdir(parents=True, exist_ok=True)

    dynamic_env = get_synthetic_env(version=version)
    print("env: {:}".format(dynamic_env))
    print("oracle_map: {:}".format(dynamic_env.oracle_map))
    allxs, allys = [], []
    for idx, (timestamp, (allx, ally)) in enumerate(tqdm(dynamic_env, ncols=50)):
        allxs.append(allx)
        allys.append(ally)
    if dynamic_env.meta_info["task"] == "regression":
        allxs, allys = torch.cat(allxs).view(-1), torch.cat(allys).view(-1)
        print(
            "x - min={:.3f}, max={:.3f}".format(allxs.min().item(), allxs.max().item())
        )
        print(
            "y - min={:.3f}, max={:.3f}".format(allys.min().item(), allys.max().item())
        )
    elif dynamic_env.meta_info["task"] == "classification":
        allxs = torch.cat(allxs)
        print(
            "x[0] - min={:.3f}, max={:.3f}".format(
                allxs[:, 0].min().item(), allxs[:, 0].max().item()
            )
        )
        print(
            "x[1] - min={:.3f}, max={:.3f}".format(
                allxs[:, 1].min().item(), allxs[:, 1].max().item()
            )
        )
    else:
        raise ValueError("Unknown task".format(dynamic_env.meta_info["task"]))

    for idx, (timestamp, (allx, ally)) in enumerate(tqdm(dynamic_env, ncols=50)):
        dpi, width, height = 30, 1800, 1400
        figsize = width / float(dpi), height / float(dpi)
        LabelSize, LegendFontsize, font_gap = 80, 80, 5
        fig = plt.figure(figsize=figsize)

        cur_ax = fig.add_subplot(1, 1, 1)
        if dynamic_env.meta_info["task"] == "regression":
            allx, ally = allx[:, 0].numpy(), ally[:, 0].numpy()
            plot_scatter(
                cur_ax, allx, ally, "k", 0.99, (15, 1.5), "timestamp={:05d}".format(idx)
            )
            cur_ax.set_xlim(round(allxs.min().item(), 1), round(allxs.max().item(), 1))
            cur_ax.set_ylim(round(allys.min().item(), 1), round(allys.max().item(), 1))
        elif dynamic_env.meta_info["task"] == "classification":
            positive, negative = ally == 1, ally == 0
            # plot_scatter(cur_ax, [1], [1], "k", 0.1, 1, "timestamp={:05d}".format(idx))
            plot_scatter(
                cur_ax,
                allx[positive, 0],
                allx[positive, 1],
                "r",
                0.99,
                (20, 10),
                "positive",
            )
            plot_scatter(
                cur_ax,
                allx[negative, 0],
                allx[negative, 1],
                "g",
                0.99,
                (20, 10),
                "negative",
            )
            cur_ax.set_xlim(
                round(allxs[:, 0].min().item(), 1), round(allxs[:, 0].max().item(), 1)
            )
            cur_ax.set_ylim(
                round(allxs[:, 1].min().item(), 1), round(allxs[:, 1].max().item(), 1)
            )
        else:
            raise ValueError("Unknown task".format(dynamic_env.meta_info["task"]))

        cur_ax.set_xlabel("X", fontsize=LabelSize)
        cur_ax.set_ylabel("Y", rotation=0, fontsize=LabelSize)
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
            tick.label.set_rotation(10)
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
        cur_ax.legend(loc=1, fontsize=LegendFontsize)
        pdf_save_path = (
            save_dir
            / "pdf-{:}".format(version)
            / "v{:}-{:05d}.pdf".format(version, idx)
        )
        fig.savefig(str(pdf_save_path), dpi=dpi, bbox_inches="tight", format="pdf")
        png_save_path = (
            save_dir
            / "png-{:}".format(version)
            / "v{:}-{:05d}.png".format(version, idx)
        )
        fig.savefig(str(png_save_path), dpi=dpi, bbox_inches="tight", format="png")
        plt.close("all")
    save_dir = save_dir.resolve()
    base_cmd = "ffmpeg -y -i {xdir}/v{version}-%05d.png -vf scale=1800:1400 -pix_fmt yuv420p -vb 5000k".format(
        xdir=save_dir / "png-{:}".format(version), version=version
    )
    print(base_cmd)
    os.system("{:} {xdir}/env-{ver}.mp4".format(base_cmd, xdir=save_dir, ver=version))
    os.system("{:} {xdir}/env-{ver}.webm".format(base_cmd, xdir=save_dir, ver=version))


def compare_algs(save_dir, version, alg_dir="./outputs/GeMOSA-synthetic"):
    save_dir = Path(str(save_dir))
    for substr in ("pdf", "png"):
        sub_save_dir = save_dir / substr
        sub_save_dir.mkdir(parents=True, exist_ok=True)

    dpi, width, height = 30, 3200, 2000
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize, font_gap = 80, 80, 5

    dynamic_env = get_synthetic_env(mode=None, version=version)
    allxs, allys = [], []
    for idx, (timestamp, (allx, ally)) in enumerate(tqdm(dynamic_env, ncols=50)):
        allxs.append(allx)
        allys.append(ally)
    allxs, allys = torch.cat(allxs).view(-1), torch.cat(allys).view(-1)

    alg_name2dir = OrderedDict()
    # alg_name2dir["Supervised Learning (History Data)"] = "use-all-past-data"
    # alg_name2dir["MAML"] = "use-maml-s1"
    # alg_name2dir["LFNA (fix init)"] = "lfna-fix-init"
    if version == "v1":
        # alg_name2dir["Optimal"] = "use-same-timestamp"
        alg_name2dir[
            "GMOA"
        ] = "lfna-battle-bs128-d16_16_16-s16-lr0.002-wd1e-05-e10000-envv1"
    else:
        raise ValueError("Invalid version: {:}".format(version))
    alg_name2all_containers = OrderedDict()
    for idx_alg, (alg, xdir) in enumerate(alg_name2dir.items()):
        ckp_path = Path(alg_dir) / str(xdir) / "final-ckp.pth"
        xdata = torch.load(ckp_path, map_location="cpu")
        alg_name2all_containers[alg] = xdata["w_containers"]
    # load the basic model
    model = get_model(
        dict(model_type="norm_mlp"),
        input_dim=1,
        output_dim=1,
        hidden_dims=[16] * 2,
        act_cls="gelu",
        norm_cls="layer_norm_1d",
    )

    alg2xs, alg2ys = defaultdict(list), defaultdict(list)
    colors = ["r", "g", "b", "m", "y"]

    linewidths, skip = 10, 5
    for idx, (timestamp, (ori_allx, ori_ally)) in enumerate(
        tqdm(dynamic_env, ncols=50)
    ):
        if idx <= skip:
            continue
        fig = plt.figure(figsize=figsize)
        cur_ax = fig.add_subplot(2, 1, 1)

        # the data
        allx, ally = ori_allx[:, 0].numpy(), ori_ally[:, 0].numpy()
        plot_scatter(cur_ax, allx, ally, "k", 0.99, linewidths, "Raw Data")

        for idx_alg, (alg, xdir) in enumerate(alg_name2dir.items()):
            with torch.no_grad():
                predicts = model.forward_with_container(
                    ori_allx, alg_name2all_containers[alg][idx]
                )
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
        cur_ax.set_xlim(round(allxs.min().item(), 1), round(allxs.max().item(), 1))
        cur_ax.set_ylim(round(allys.min().item(), 1), round(allys.max().item(), 1))
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

        pdf_save_path = save_dir / "pdf" / "v{:}-{:05d}.pdf".format(version, idx - skip)
        fig.savefig(str(pdf_save_path), dpi=dpi, bbox_inches="tight", format="pdf")
        png_save_path = save_dir / "png" / "v{:}-{:05d}.png".format(version, idx - skip)
        fig.savefig(str(png_save_path), dpi=dpi, bbox_inches="tight", format="png")
        plt.close("all")
    save_dir = save_dir.resolve()
    base_cmd = "ffmpeg -y -i {xdir}/v{ver}-%05d.png -vf scale={w}:{h} -pix_fmt yuv420p -vb 5000k".format(
        xdir=save_dir / "png", w=width, h=height, ver=version
    )
    os.system(
        "{:} {xdir}/com-alg-{ver}.mp4".format(base_cmd, xdir=save_dir, ver=version)
    )
    os.system(
        "{:} {xdir}/com-alg-{ver}.webm".format(base_cmd, xdir=save_dir, ver=version)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visualize synthetic data.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/vis-synthetic",
        help="The save directory.",
    )
    parser.add_argument(
        "--env_version",
        type=str,
        required=True,
        help="The synthetic enviornment version.",
    )
    args = parser.parse_args()

    visualize_env(os.path.join(args.save_dir, "vis-env"), args.env_version)
    # visualize_env(os.path.join(args.save_dir, "vis-env"), "v2")
    # compare_algs(os.path.join(args.save_dir, "compare-alg"), args.env_version)
    # compare_cl(os.path.join(args.save_dir, "compare-cl"))
