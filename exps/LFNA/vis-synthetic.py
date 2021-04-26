#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
############################################################################
# CUDA_VISIBLE_DEVICES=0 python exps/LFNA/vis-synthetic.py                 #
############################################################################
import os, sys, copy, random
import torch
import numpy as np
import argparse
from collections import OrderedDict
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


from datasets import ConstantGenerator, SinGenerator, SyntheticDEnv
from datasets import DynamicQuadraticFunc
from datasets.synthetic_example import create_example_v1

from utils.temp_sync import optimize_fn, evaluate_fn


def draw_multi_fig(save_dir, timestamp, scatter_list, fig_title=None):
    save_path = save_dir / "{:04d}".format(timestamp)
    # print('Plot the figure at timestamp-{:} into {:}'.format(timestamp, save_path))
    dpi, width, height = 40, 2000, 1300
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize, font_gap = 80, 80, 5

    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=LegendFontsize)

    for idx, scatter_dict in enumerate(scatter_list):
        cur_ax = fig.add_subplot(len(scatter_list), 1, idx + 1)
        cur_ax.scatter(
            scatter_dict["xaxis"],
            scatter_dict["yaxis"],
            color=scatter_dict["color"],
            s=scatter_dict["s"],
            alpha=scatter_dict["alpha"],
            label=scatter_dict["label"],
        )
        cur_ax.set_xlabel("X", fontsize=LabelSize)
        cur_ax.set_ylabel("f(X)", rotation=0, fontsize=LabelSize)
        cur_ax.set_xlim(scatter_dict["xlim"][0], scatter_dict["xlim"][1])
        cur_ax.set_ylim(scatter_dict["ylim"][0], scatter_dict["ylim"][1])
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)
            tick.label.set_rotation(10)
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(LabelSize - font_gap)

    plt.legend(loc=1, fontsize=LegendFontsize)
    fig.savefig(str(save_path) + ".pdf", dpi=dpi, bbox_inches="tight", format="pdf")
    fig.savefig(str(save_path) + ".png", dpi=dpi, bbox_inches="tight", format="png")
    plt.close("all")


def compare_cl(save_dir):
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    dynamic_env, function = create_example_v1(100, num_per_task=1000)

    additional_xaxis = np.arange(-6, 6, 0.2)
    models = dict()

    cl_function = copy.deepcopy(function)
    cl_function.set_timestamp(0)
    cl_xaxis_all = None

    for idx, (timestamp, dataset) in enumerate(tqdm(dynamic_env, ncols=50)):
        xaxis_all = dataset[:, 0].numpy()
        # xaxis_all = np.concatenate((additional_xaxis, xaxis_all))
        # compute the ground truth
        function.set_timestamp(timestamp)
        yaxis_all = function.noise_call(xaxis_all)

        # create CL data
        if cl_xaxis_all is None:
            cl_xaxis_all = xaxis_all
        else:
            cl_xaxis_all = np.concatenate((cl_xaxis_all, xaxis_all + timestamp * 0.2))
        cl_yaxis_all = cl_function(cl_xaxis_all)

        scatter_list = []
        scatter_list.append(
            {
                "xaxis": xaxis_all,
                "yaxis": yaxis_all,
                "color": "k",
                "s": 10,
                "alpha": 0.99,
                "xlim": (-6, 6),
                "ylim": (-40, 40),
                "label": "LFNA",
            }
        )

        scatter_list.append(
            {
                "xaxis": cl_xaxis_all,
                "yaxis": cl_yaxis_all,
                "color": "r",
                "s": 10,
                "xlim": (-6, 6 + timestamp * 0.2),
                "ylim": (-200, 40),
                "alpha": 0.99,
                "label": "Continual Learning",
            }
        )

        draw_multi_fig(
            save_dir, timestamp, scatter_list, "Timestamp={:03d}".format(timestamp)
        )
    print("Save all figures into {:}".format(save_dir))
    save_dir = save_dir.resolve()
    cmd = "ffmpeg -y -i {xdir}/%04d.png -pix_fmt yuv420p -vf fps=2 -vf scale=1500:1000 -vb 5000k {xdir}/vis.mp4".format(
        xdir=save_dir
    )
    os.system(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visualize synthetic data.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/vis-synthetic",
        help="The save directory.",
    )
    args = parser.parse_args()

    compare_cl(os.path.join(args.save_dir, "compare-cl"))
