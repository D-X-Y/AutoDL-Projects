##############################################################################
# NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08                          #
##############################################################################
# Usage: python exps/NATS-Bench/tss-file-manager.py --mode check             #
##############################################################################
import os, sys, time, torch, argparse
from typing import List, Text, Dict, Any
from shutil import copyfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from xautodl.config_utils import dict2config, load_config
from xautodl.procedures import bench_evaluate_for_seed
from xautodl.procedures import get_machine_info
from xautodl.datasets import get_datasets
from xautodl.log_utils import Logger, AverageMeter, time_string, convert_secs2time


def obtain_valid_ckp(save_dir: Text, total: int, possible_seeds: List[int]):
    seed2ckps = defaultdict(list)
    miss2ckps = defaultdict(list)
    for i in range(total):
        for seed in possible_seeds:
            path = os.path.join(save_dir, "arch-{:06d}-seed-{:04d}.pth".format(i, seed))
            if os.path.exists(path):
                seed2ckps[seed].append(i)
            else:
                miss2ckps[seed].append(i)
    for seed, xlist in seed2ckps.items():
        print(
            "[{:}] [seed={:}] has {:5d}/{:5d} | miss {:5d}/{:5d}".format(
                save_dir, seed, len(xlist), total, total - len(xlist), total
            )
        )
    return dict(seed2ckps), dict(miss2ckps)


def copy_data(source_dir, target_dir, meta_path):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    miss2ckps = torch.load(meta_path)["miss2ckps"]
    s2t = {}
    for seed, xlist in miss2ckps.items():
        for i in xlist:
            file_name = "arch-{:06d}-seed-{:04d}.pth".format(i, seed)
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            if os.path.exists(source_path):
                s2t[source_path] = target_path
    print(
        "Map from {:} to {:}, find {:} missed ckps.".format(
            source_dir, target_dir, len(s2t)
        )
    )
    for s, t in s2t.items():
        copyfile(s, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NATS-Bench (topology search space) file manager.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["check", "copy"],
        help="The script mode.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/NATS-Bench-topology",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument("--check_N", type=int, default=15625, help="For safety.")
    # use for train the model
    args = parser.parse_args()
    possible_configs = ["12", "200"]
    possible_seedss = [[111, 777], [777, 888, 999]]
    if args.mode == "check":
        for config, possible_seeds in zip(possible_configs, possible_seedss):
            cur_save_dir = "{:}/raw-data-{:}".format(args.save_dir, config)
            seed2ckps, miss2ckps = obtain_valid_ckp(
                cur_save_dir, args.check_N, possible_seeds
            )
            torch.save(
                dict(seed2ckps=seed2ckps, miss2ckps=miss2ckps),
                "{:}/meta-{:}.pth".format(args.save_dir, config),
            )
    elif args.mode == "copy":
        for config in possible_configs:
            cur_save_dir = "{:}/raw-data-{:}".format(args.save_dir, config)
            cur_copy_dir = "{:}/copy-{:}".format(args.save_dir, config)
            cur_meta_path = "{:}/meta-{:}.pth".format(args.save_dir, config)
            if os.path.exists(cur_meta_path):
                copy_data(cur_save_dir, cur_copy_dir, cur_meta_path)
            else:
                print("Do not find : {:}".format(cur_meta_path))
    else:
        raise ValueError("invalid mode : {:}".format(args.mode))
