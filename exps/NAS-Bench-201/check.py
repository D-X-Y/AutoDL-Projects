#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
# python exps/NAS-Bench-201/check.py --base_str C16-N5-LESS
#####################################################
import sys, time, argparse, collections
import torch
from pathlib import Path
from collections import defaultdict

from xautodl.log_utils import AverageMeter, time_string, convert_secs2time


def check_files(save_dir, meta_file, basestr):
    meta_infos = torch.load(meta_file, map_location="cpu")
    meta_archs = meta_infos["archs"]
    meta_num_archs = meta_infos["total"]
    assert meta_num_archs == len(
        meta_archs
    ), "invalid number of archs : {:} vs {:}".format(meta_num_archs, len(meta_archs))

    sub_model_dirs = sorted(list(save_dir.glob("*-*-{:}".format(basestr))))
    print(
        "{:} find {:} directories used to save checkpoints".format(
            time_string(), len(sub_model_dirs)
        )
    )

    subdir2archs, num_evaluated_arch = collections.OrderedDict(), 0
    num_seeds = defaultdict(lambda: 0)
    for index, sub_dir in enumerate(sub_model_dirs):
        xcheckpoints = list(sub_dir.glob("arch-*-seed-*.pth"))
        # xcheckpoints = list(sub_dir.glob('arch-*-seed-0777.pth')) + list(sub_dir.glob('arch-*-seed-0888.pth')) + list(sub_dir.glob('arch-*-seed-0999.pth'))
        arch_indexes = set()
        for checkpoint in xcheckpoints:
            temp_names = checkpoint.name.split("-")
            assert (
                len(temp_names) == 4
                and temp_names[0] == "arch"
                and temp_names[2] == "seed"
            ), "invalid checkpoint name : {:}".format(checkpoint.name)
            arch_indexes.add(temp_names[1])
        subdir2archs[sub_dir] = sorted(list(arch_indexes))
        num_evaluated_arch += len(arch_indexes)
        # count number of seeds for each architecture
        for arch_index in arch_indexes:
            num_seeds[
                len(list(sub_dir.glob("arch-{:}-seed-*.pth".format(arch_index))))
            ] += 1
    print(
        "There are {:5d} architectures that have been evaluated ({:} in total, {:} ckps in total).".format(
            num_evaluated_arch, meta_num_archs, sum(k * v for k, v in num_seeds.items())
        )
    )
    for key in sorted(list(num_seeds.keys())):
        print(
            "There are {:5d} architectures that are evaluated {:} times.".format(
                num_seeds[key], key
            )
        )

    dir2ckps, dir2ckp_exists = dict(), dict()
    start_time, epoch_time = time.time(), AverageMeter()
    for IDX, (sub_dir, arch_indexes) in enumerate(subdir2archs.items()):
        if basestr == "C16-N5":
            seeds = [777, 888, 999]
        elif basestr == "C16-N5-LESS":
            seeds = [111, 777]
        else:
            raise ValueError("Invalid base str : {:}".format(basestr))
        numrs = defaultdict(lambda: 0)
        all_checkpoints, all_ckp_exists = [], []
        for arch_index in arch_indexes:
            checkpoints = [
                "arch-{:}-seed-{:04d}.pth".format(arch_index, seed) for seed in seeds
            ]
            ckp_exists = [(sub_dir / x).exists() for x in checkpoints]
            arch_index = int(arch_index)
            assert (
                0 <= arch_index < len(meta_archs)
            ), "invalid arch-index {:} (not found in meta_archs)".format(arch_index)
            all_checkpoints += checkpoints
            all_ckp_exists += ckp_exists
            numrs[sum(ckp_exists)] += 1
        dir2ckps[str(sub_dir)] = all_checkpoints
        dir2ckp_exists[str(sub_dir)] = all_ckp_exists
        # measure time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        numrstr = ", ".join(
            ["{:}: {:03d}".format(x, numrs[x]) for x in sorted(numrs.keys())]
        )
        print(
            "{:} load [{:2d}/{:2d}] [{:03d} archs] [{:04d}->{:04d} ckps] {:} done, need {:}. {:}".format(
                time_string(),
                IDX + 1,
                len(subdir2archs),
                len(arch_indexes),
                len(all_checkpoints),
                sum(all_ckp_exists),
                sub_dir,
                convert_secs2time(epoch_time.avg * (len(subdir2archs) - IDX - 1), True),
                numrstr,
            )
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NAS Benchmark 201",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="./output/NAS-BENCH-201-4",
        help="The base-name of folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="./output/NAS-BENCH-201-4/meta-node-4.pth",
        help="The meta file path.",
    )
    parser.add_argument(
        "--base_str", type=str, default="C16-N5", help="The basic string."
    )
    args = parser.parse_args()

    save_dir = Path(args.base_save_dir)
    meta_path = Path(args.meta_path)
    assert save_dir.exists(), "invalid save dir path : {:}".format(save_dir)
    assert meta_path.exists(), "invalid saved meta path : {:}".format(meta_path)
    print("check NAS-Bench-201 in {:}".format(save_dir))

    check_files(save_dir, meta_path, args.base_str)
