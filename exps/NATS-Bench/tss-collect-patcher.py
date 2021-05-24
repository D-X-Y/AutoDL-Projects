##############################################################################
# NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08                          #
##############################################################################
# This file is used to re-orangize all checkpoints (created by main-tss.py)  #
# into a single benchmark file. Besides, for each trial, we will merge the   #
# information of all its trials into a single file.                          #
#                                                                            #
# Usage:                                                                     #
# python exps/NATS-Bench/tss-collect-patcher.py                              #
##############################################################################
import os, re, sys, time, shutil, random, argparse, collections
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, Any, Text, List

from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.config_utils import load_config, dict2config
from xautodl.datasets import get_datasets
from xautodl.models import CellStructure, get_cell_based_tiny_net, get_search_spaces
from xautodl.procedures import (
    bench_pure_evaluate as pure_evaluate,
    get_nas_bench_loaders,
)
from xautodl.utils import get_md5_file

from nats_bench import pickle_save, pickle_load, ArchResults, ResultsCount
from nas_201_api import NASBench201API


NATS_TSS_BASE_NAME = "NATS-tss-v1_0"  # 2020.08.28


def simplify(save_dir, save_name, nets, total, sup_config):
    hps, seeds = ["12", "200"], set()
    for hp in hps:
        sub_save_dir = save_dir / "raw-data-{:}".format(hp)
        ckps = sorted(list(sub_save_dir.glob("arch-*-seed-*.pth")))
        seed2names = defaultdict(list)
        for ckp in ckps:
            parts = re.split("-|\.", ckp.name)
            seed2names[parts[3]].append(ckp.name)
        print("DIR : {:}".format(sub_save_dir))
        nums = []
        for seed, xlist in seed2names.items():
            seeds.add(seed)
            nums.append(len(xlist))
            print("  [seed={:}] there are {:} checkpoints.".format(seed, len(xlist)))
        assert (
            len(nets) == total == max(nums)
        ), "there are some missed files : {:} vs {:}".format(max(nums), total)
    print("{:} start simplify the checkpoint.".format(time_string()))

    datasets = ("cifar10-valid", "cifar10", "cifar100", "ImageNet16-120")

    # Create the directory to save the processed data
    # full_save_dir contains all benchmark files with trained weights.
    # simplify_save_dir contains all benchmark files without trained weights.
    full_save_dir = save_dir / (save_name + "-FULL")
    simple_save_dir = save_dir / (save_name + "-SIMPLIFY")
    full_save_dir.mkdir(parents=True, exist_ok=True)
    simple_save_dir.mkdir(parents=True, exist_ok=True)
    # all data in memory
    arch2infos, evaluated_indexes = dict(), set()
    end_time, arch_time = time.time(), AverageMeter()
    # save the meta information
    for index in tqdm(range(total)):
        arch_str = nets[index]
        hp2info = OrderedDict()

        simple_save_path = simple_save_dir / "{:06d}.pickle".format(index)

        arch2infos[index] = pickle_load(simple_save_path)
        evaluated_indexes.add(index)

        # measure elapsed time
        arch_time.update(time.time() - end_time)
        end_time = time.time()
        need_time = "{:}".format(
            convert_secs2time(arch_time.avg * (total - index - 1), True)
        )
        # print('{:} {:06d}/{:06d} : still need {:}'.format(time_string(), index, total, need_time))
    print("{:} {:} done.".format(time_string(), save_name))
    final_infos = {
        "meta_archs": nets,
        "total_archs": total,
        "arch2infos": arch2infos,
        "evaluated_indexes": evaluated_indexes,
    }
    save_file_name = save_dir / "{:}.pickle".format(save_name)
    pickle_save(final_infos, str(save_file_name))
    # move the benchmark file to a new path
    hd5sum = get_md5_file(str(save_file_name) + ".pbz2")
    hd5_file_name = save_dir / "{:}-{:}.pickle.pbz2".format(NATS_TSS_BASE_NAME, hd5sum)
    shutil.move(str(save_file_name) + ".pbz2", hd5_file_name)
    print(
        "Save {:} / {:} architecture results into {:} -> {:}.".format(
            len(evaluated_indexes), total, save_file_name, hd5_file_name
        )
    )
    # move the directory to a new path
    hd5_full_save_dir = save_dir / "{:}-{:}-full".format(NATS_TSS_BASE_NAME, hd5sum)
    hd5_simple_save_dir = save_dir / "{:}-{:}-simple".format(NATS_TSS_BASE_NAME, hd5sum)
    shutil.move(full_save_dir, hd5_full_save_dir)
    shutil.move(simple_save_dir, hd5_simple_save_dir)


def traverse_net(max_node):
    aa_nas_bench_ss = get_search_spaces("cell", "nats-bench")
    archs = CellStructure.gen_all(aa_nas_bench_ss, max_node, False)
    print(
        "There are {:} archs vs {:}.".format(
            len(archs), len(aa_nas_bench_ss) ** ((max_node - 1) * max_node / 2)
        )
    )

    random.seed(88)  # please do not change this line for reproducibility
    random.shuffle(archs)
    assert (
        archs[0].tostr()
        == "|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|"
    ), "please check the 0-th architecture : {:}".format(archs[0])
    assert (
        archs[9].tostr()
        == "|avg_pool_3x3~0|+|none~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|"
    ), "please check the 9-th architecture : {:}".format(archs[9])
    assert (
        archs[123].tostr()
        == "|avg_pool_3x3~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|none~0|avg_pool_3x3~1|nor_conv_3x3~2|"
    ), "please check the 123-th architecture : {:}".format(archs[123])
    return [x.tostr() for x in archs]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NATS-Bench (topology search space)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="./output/NATS-Bench-topology",
        help="The base-name of folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--max_node", type=int, default=4, help="The maximum node in a cell."
    )
    parser.add_argument(
        "--channel", type=int, default=16, help="The number of channels."
    )
    parser.add_argument(
        "--num_cells", type=int, default=5, help="The number of cells in one stage."
    )
    parser.add_argument("--check_N", type=int, default=15625, help="For safety.")
    parser.add_argument(
        "--save_name", type=str, default="process", help="The save directory."
    )
    args = parser.parse_args()

    nets = traverse_net(args.max_node)
    if len(nets) != args.check_N:
        raise ValueError(
            "Pre-num-check failed : {:} vs {:}".format(len(nets), args.check_N)
        )

    save_dir = Path(args.base_save_dir)
    simplify(
        save_dir,
        args.save_name,
        nets,
        args.check_N,
        {"name": "infer.tiny", "channel": args.channel, "num_cells": args.num_cells},
    )
