#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
###########################################################################################################################################################
# Before run these commands, the files must be properly put.
#
# python exps/experimental/example-nas-bench.py --api_path $HOME/.torch/NAS-Bench-201-v1_1-096897.pth --archive_path $HOME/.torch/NAS-Bench-201-v1_1-archive
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

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from nas_201_api import NASBench201API
from log_utils import time_string
from models import get_cell_based_tiny_net
from utils import weight_watcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis of NAS-Bench-201")
    parser.add_argument(
        "--api_path",
        type=str,
        default=None,
        help="The path to the NAS-Bench-201 benchmark file and weight dir.",
    )
    parser.add_argument(
        "--archive_path",
        type=str,
        default=None,
        help="The path to the NAS-Bench-201 weight dir.",
    )
    args = parser.parse_args()

    meta_file = Path(args.api_path)
    weight_dir = Path(args.archive_path)
    assert meta_file.exists(), "invalid path for api : {:}".format(meta_file)
    assert (
        weight_dir.exists() and weight_dir.is_dir()
    ), "invalid path for weight dir : {:}".format(weight_dir)

    api = NASBench201API(meta_file, verbose=True)

    arch_index = 3  # query the 3-th architecture
    api.reload(weight_dir, arch_index)  # reload the data of 3-th from archive dir

    data = "cifar10"  # query the info from CIFAR-10
    config = api.get_net_config(arch_index, data)
    net = get_cell_based_tiny_net(config)
    meta_info = api.query_meta_info_by_index(
        arch_index, hp="200"
    )  # all info about this architecture
    params = meta_info.get_net_param(data, 888)

    net.load_state_dict(params)
    _, summary = weight_watcher.analyze(net, alphas=False)
    print("The summary of {:}-th architecture:\n{:}".format(arch_index, summary))
