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
# python exps/NATS-Bench/tss-collect.py                                      #
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


api = NASBench201API(
    "{:}/.torch/NAS-Bench-201-v1_0-e61699.pth".format(os.environ["HOME"])
)

NATS_TSS_BASE_NAME = "NATS-tss-v1_0"  # 2020.08.28


def create_result_count(
    used_seed: int,
    dataset: Text,
    arch_config: Dict[Text, Any],
    results: Dict[Text, Any],
    dataloader_dict: Dict[Text, Any],
) -> ResultsCount:
    xresult = ResultsCount(
        dataset,
        results["net_state_dict"],
        results["train_acc1es"],
        results["train_losses"],
        results["param"],
        results["flop"],
        arch_config,
        used_seed,
        results["total_epoch"],
        None,
    )
    net_config = dict2config(
        {
            "name": "infer.tiny",
            "C": arch_config["channel"],
            "N": arch_config["num_cells"],
            "genotype": CellStructure.str2structure(arch_config["arch_str"]),
            "num_classes": arch_config["class_num"],
        },
        None,
    )
    if "train_times" in results:  # new version
        xresult.update_train_info(
            results["train_acc1es"],
            results["train_acc5es"],
            results["train_losses"],
            results["train_times"],
        )
        xresult.update_eval(
            results["valid_acc1es"], results["valid_losses"], results["valid_times"]
        )
    else:
        network = get_cell_based_tiny_net(net_config)
        network.load_state_dict(xresult.get_net_param())
        if dataset == "cifar10-valid":
            xresult.update_OLD_eval(
                "x-valid", results["valid_acc1es"], results["valid_losses"]
            )
            loss, top1, top5, latencies = pure_evaluate(
                dataloader_dict["{:}@{:}".format("cifar10", "test")], network.cuda()
            )
            xresult.update_OLD_eval(
                "ori-test",
                {results["total_epoch"] - 1: top1},
                {results["total_epoch"] - 1: loss},
            )
            xresult.update_latency(latencies)
        elif dataset == "cifar10":
            xresult.update_OLD_eval(
                "ori-test", results["valid_acc1es"], results["valid_losses"]
            )
            loss, top1, top5, latencies = pure_evaluate(
                dataloader_dict["{:}@{:}".format(dataset, "test")], network.cuda()
            )
            xresult.update_latency(latencies)
        elif dataset == "cifar100" or dataset == "ImageNet16-120":
            xresult.update_OLD_eval(
                "ori-test", results["valid_acc1es"], results["valid_losses"]
            )
            loss, top1, top5, latencies = pure_evaluate(
                dataloader_dict["{:}@{:}".format(dataset, "valid")], network.cuda()
            )
            xresult.update_OLD_eval(
                "x-valid",
                {results["total_epoch"] - 1: top1},
                {results["total_epoch"] - 1: loss},
            )
            loss, top1, top5, latencies = pure_evaluate(
                dataloader_dict["{:}@{:}".format(dataset, "test")], network.cuda()
            )
            xresult.update_OLD_eval(
                "x-test",
                {results["total_epoch"] - 1: top1},
                {results["total_epoch"] - 1: loss},
            )
            xresult.update_latency(latencies)
        else:
            raise ValueError("invalid dataset name : {:}".format(dataset))
    return xresult


def account_one_arch(arch_index, arch_str, checkpoints, datasets, dataloader_dict):
    information = ArchResults(arch_index, arch_str)

    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        used_seed = checkpoint_path.name.split("-")[-1].split(".")[0]
        ok_dataset = 0
        for dataset in datasets:
            if dataset not in checkpoint:
                print(
                    "Can not find {:} in arch-{:} from {:}".format(
                        dataset, arch_index, checkpoint_path
                    )
                )
                continue
            else:
                ok_dataset += 1
            results = checkpoint[dataset]
            assert results[
                "finish-train"
            ], "This {:} arch seed={:} does not finish train on {:} ::: {:}".format(
                arch_index, used_seed, dataset, checkpoint_path
            )
            arch_config = {
                "channel": results["channel"],
                "num_cells": results["num_cells"],
                "arch_str": arch_str,
                "class_num": results["config"]["class_num"],
            }

            xresult = create_result_count(
                used_seed, dataset, arch_config, results, dataloader_dict
            )
            information.update(dataset, int(used_seed), xresult)
        if ok_dataset == 0:
            raise ValueError("{:} does not find any data".format(checkpoint_path))
    return information


def correct_time_related_info(arch_index: int, arch_infos: Dict[Text, ArchResults]):
    # calibrate the latency based on NAS-Bench-201-v1_0-e61699.pth
    cifar010_latency = (
        api.get_latency(arch_index, "cifar10-valid", hp="200")
        + api.get_latency(arch_index, "cifar10", hp="200")
    ) / 2
    cifar100_latency = api.get_latency(arch_index, "cifar100", hp="200")
    image_latency = api.get_latency(arch_index, "ImageNet16-120", hp="200")
    for hp, arch_info in arch_infos.items():
        arch_info.reset_latency("cifar10-valid", None, cifar010_latency)
        arch_info.reset_latency("cifar10", None, cifar010_latency)
        arch_info.reset_latency("cifar100", None, cifar100_latency)
        arch_info.reset_latency("ImageNet16-120", None, image_latency)

    train_per_epoch_time = list(
        arch_infos["12"].query("cifar10-valid", 777).train_times.values()
    )
    train_per_epoch_time = sum(train_per_epoch_time) / len(train_per_epoch_time)
    eval_ori_test_time, eval_x_valid_time = [], []
    for key, value in arch_infos["12"].query("cifar10-valid", 777).eval_times.items():
        if key.startswith("ori-test@"):
            eval_ori_test_time.append(value)
        elif key.startswith("x-valid@"):
            eval_x_valid_time.append(value)
        else:
            raise ValueError("-- {:} --".format(key))
    eval_ori_test_time, eval_x_valid_time = float(np.mean(eval_ori_test_time)), float(
        np.mean(eval_x_valid_time)
    )
    nums = {
        "ImageNet16-120-train": 151700,
        "ImageNet16-120-valid": 3000,
        "ImageNet16-120-test": 6000,
        "cifar10-valid-train": 25000,
        "cifar10-valid-valid": 25000,
        "cifar10-train": 50000,
        "cifar10-test": 10000,
        "cifar100-train": 50000,
        "cifar100-test": 10000,
        "cifar100-valid": 5000,
    }
    eval_per_sample = (eval_ori_test_time + eval_x_valid_time) / (
        nums["cifar10-valid-valid"] + nums["cifar10-test"]
    )
    for hp, arch_info in arch_infos.items():
        arch_info.reset_pseudo_train_times(
            "cifar10-valid",
            None,
            train_per_epoch_time
            / nums["cifar10-valid-train"]
            * nums["cifar10-valid-train"],
        )
        arch_info.reset_pseudo_train_times(
            "cifar10",
            None,
            train_per_epoch_time / nums["cifar10-valid-train"] * nums["cifar10-train"],
        )
        arch_info.reset_pseudo_train_times(
            "cifar100",
            None,
            train_per_epoch_time / nums["cifar10-valid-train"] * nums["cifar100-train"],
        )
        arch_info.reset_pseudo_train_times(
            "ImageNet16-120",
            None,
            train_per_epoch_time
            / nums["cifar10-valid-train"]
            * nums["ImageNet16-120-train"],
        )
        arch_info.reset_pseudo_eval_times(
            "cifar10-valid",
            None,
            "x-valid",
            eval_per_sample * nums["cifar10-valid-valid"],
        )
        arch_info.reset_pseudo_eval_times(
            "cifar10-valid", None, "ori-test", eval_per_sample * nums["cifar10-test"]
        )
        arch_info.reset_pseudo_eval_times(
            "cifar10", None, "ori-test", eval_per_sample * nums["cifar10-test"]
        )
        arch_info.reset_pseudo_eval_times(
            "cifar100", None, "x-valid", eval_per_sample * nums["cifar100-valid"]
        )
        arch_info.reset_pseudo_eval_times(
            "cifar100", None, "x-test", eval_per_sample * nums["cifar100-valid"]
        )
        arch_info.reset_pseudo_eval_times(
            "cifar100", None, "ori-test", eval_per_sample * nums["cifar100-test"]
        )
        arch_info.reset_pseudo_eval_times(
            "ImageNet16-120",
            None,
            "x-valid",
            eval_per_sample * nums["ImageNet16-120-valid"],
        )
        arch_info.reset_pseudo_eval_times(
            "ImageNet16-120",
            None,
            "x-test",
            eval_per_sample * nums["ImageNet16-120-valid"],
        )
        arch_info.reset_pseudo_eval_times(
            "ImageNet16-120",
            None,
            "ori-test",
            eval_per_sample * nums["ImageNet16-120-test"],
        )
    return arch_infos


def simplify(save_dir, save_name, nets, total, sup_config):
    dataloader_dict = get_nas_bench_loaders(6)
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
    temp_final_infos = {
        "meta_archs": nets,
        "total_archs": total,
        "arch2infos": None,
        "evaluated_indexes": set(),
    }
    pickle_save(temp_final_infos, str(full_save_dir / "meta.pickle"))
    pickle_save(temp_final_infos, str(simple_save_dir / "meta.pickle"))

    for index in tqdm(range(total)):
        arch_str = nets[index]
        hp2info = OrderedDict()

        full_save_path = full_save_dir / "{:06d}.pickle".format(index)
        simple_save_path = simple_save_dir / "{:06d}.pickle".format(index)
        for hp in hps:
            sub_save_dir = save_dir / "raw-data-{:}".format(hp)
            ckps = [
                sub_save_dir / "arch-{:06d}-seed-{:}.pth".format(index, seed)
                for seed in seeds
            ]
            ckps = [x for x in ckps if x.exists()]
            if len(ckps) == 0:
                raise ValueError("Invalid data : index={:}, hp={:}".format(index, hp))

            arch_info = account_one_arch(
                index, arch_str, ckps, datasets, dataloader_dict
            )
            hp2info[hp] = arch_info

        hp2info = correct_time_related_info(index, hp2info)
        evaluated_indexes.add(index)

        to_save_data = OrderedDict(
            {"12": hp2info["12"].state_dict(), "200": hp2info["200"].state_dict()}
        )
        pickle_save(to_save_data, str(full_save_path))

        for hp in hps:
            hp2info[hp].clear_params()
        to_save_data = OrderedDict(
            {"12": hp2info["12"].state_dict(), "200": hp2info["200"].state_dict()}
        )
        pickle_save(to_save_data, str(simple_save_path))
        arch2infos[index] = to_save_data
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
    # save the meta information for simple and full
    # final_infos['arch2infos'] = None
    # final_infos['evaluated_indexes'] = set()


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
