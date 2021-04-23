#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.02 #
#####################################################
# python exps/synthetic/baseline.py                 #
#####################################################
import os, sys, copy
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


def draw_fig(save_dir, timestamp, scatter_list):
    save_path = save_dir / "{:04d}".format(timestamp)
    # print('Plot the figure at timestamp-{:} into {:}'.format(timestamp, save_path))
    dpi, width, height = 40, 1500, 1500
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize, font_gap = 80, 80, 5

    fig = plt.figure(figsize=figsize)

    cur_ax = fig.add_subplot(1, 1, 1)
    for scatter_dict in scatter_list:
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
    cur_ax.set_xlim(-6, 6)
    cur_ax.set_ylim(-40, 40)
    for tick in cur_ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(LabelSize - font_gap)
        tick.label.set_rotation(10)
    for tick in cur_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(LabelSize - font_gap)

    plt.legend(loc=1, fontsize=LegendFontsize)
    fig.savefig(str(save_path) + ".pdf", dpi=dpi, bbox_inches="tight", format="pdf")
    fig.savefig(str(save_path) + ".png", dpi=dpi, bbox_inches="tight", format="png")
    plt.close("all")


def main(save_dir):
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    dynamic_env, function = create_example_v1(100, num_per_task=500)

    additional_xaxis = np.arange(-6, 6, 0.1)
    for timestamp, dataset in tqdm(dynamic_env, ncols=50):
        num = dataset.shape[0]
        xaxis = dataset[:, 0].numpy()
        # compute the ground truth
        function.set_timestamp(timestamp)
        yaxis = function(xaxis)
        # xaxis = np.concatenate((additional_xaxis, xaxis))
        # the first plot
        scatter_list = []
        scatter_list.append(
            {
                "xaxis": xaxis,
                "yaxis": yaxis,
                "color": "k",
                "s": 10,
                "alpha": 0.99,
                "label": "Timestamp={:02d}".format(timestamp),
            }
        )
        draw_fig(save_dir, timestamp, scatter_list)
    print("Save all figures into {:}".format(save_dir))
    save_dir = save_dir.resolve()
    cmd = "ffmpeg -y -i {xdir}/%04d.png -pix_fmt yuv420p -vf fps=2 -vf scale=1000:1000 -vb 5000k {xdir}/vis.mp4".format(
        xdir=save_dir
    )
    os.system(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Baseline")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/vis-synthetic",
        help="The save directory.",
    )
    args = parser.parse_args()

    main(args.save_dir)
