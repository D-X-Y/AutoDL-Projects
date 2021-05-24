#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
import os, sys, time, argparse, collections
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, Any, Text, List

from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.config_utils import dict2config

# NAS-Bench-201 related module or function
from xautodl.models import CellStructure, get_cell_based_tiny_net
from xautodl.procedures import (
    bench_pure_evaluate as pure_evaluate,
    get_nas_bench_loaders,
)
from nas_201_api import NASBench201API, ArchResults, ResultsCount

api = NASBench201API(
    "{:}/.torch/NAS-Bench-201-v1_0-e61699.pth".format(os.environ["HOME"])
)


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
    network = get_cell_based_tiny_net(net_config)
    network.load_state_dict(xresult.get_net_param())
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


def account_one_arch(
    arch_index: int,
    arch_str: Text,
    checkpoints: List[Text],
    datasets: List[Text],
    dataloader_dict: Dict[Text, Any],
) -> ArchResults:
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


def correct_time_related_info(
    arch_index: int, arch_info_full: ArchResults, arch_info_less: ArchResults
):
    # calibrate the latency based on NAS-Bench-201-v1_0-e61699.pth
    cifar010_latency = (
        api.get_latency(arch_index, "cifar10-valid", hp="200")
        + api.get_latency(arch_index, "cifar10", hp="200")
    ) / 2
    arch_info_full.reset_latency("cifar10-valid", None, cifar010_latency)
    arch_info_full.reset_latency("cifar10", None, cifar010_latency)
    arch_info_less.reset_latency("cifar10-valid", None, cifar010_latency)
    arch_info_less.reset_latency("cifar10", None, cifar010_latency)

    cifar100_latency = api.get_latency(arch_index, "cifar100", hp="200")
    arch_info_full.reset_latency("cifar100", None, cifar100_latency)
    arch_info_less.reset_latency("cifar100", None, cifar100_latency)

    image_latency = api.get_latency(arch_index, "ImageNet16-120", hp="200")
    arch_info_full.reset_latency("ImageNet16-120", None, image_latency)
    arch_info_less.reset_latency("ImageNet16-120", None, image_latency)

    train_per_epoch_time = list(
        arch_info_less.query("cifar10-valid", 777).train_times.values()
    )
    train_per_epoch_time = sum(train_per_epoch_time) / len(train_per_epoch_time)
    eval_ori_test_time, eval_x_valid_time = [], []
    for key, value in arch_info_less.query("cifar10-valid", 777).eval_times.items():
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
    for arch_info in [arch_info_less, arch_info_full]:
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
    # arch_info_full.debug_test()
    # arch_info_less.debug_test()
    return arch_info_full, arch_info_less


def simplify(save_dir, meta_file, basestr, target_dir):
    meta_infos = torch.load(meta_file, map_location="cpu")
    meta_archs = meta_infos["archs"]  # a list of architecture strings
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
        "{:} There are {:5d} architectures that have been evaluated ({:} in total).".format(
            time_string(), num_evaluated_arch, meta_num_archs
        )
    )
    for key in sorted(list(num_seeds.keys())):
        print(
            "{:} There are {:5d} architectures that are evaluated {:} times.".format(
                time_string(), num_seeds[key], key
            )
        )

    dataloader_dict = get_nas_bench_loaders(6)
    to_save_simply = save_dir / "simplifies"
    to_save_allarc = save_dir / "simplifies" / "architectures"
    if not to_save_simply.exists():
        to_save_simply.mkdir(parents=True, exist_ok=True)
    if not to_save_allarc.exists():
        to_save_allarc.mkdir(parents=True, exist_ok=True)

    assert (save_dir / target_dir) in subdir2archs, "can not find {:}".format(
        target_dir
    )
    arch2infos, datasets = {}, (
        "cifar10-valid",
        "cifar10",
        "cifar100",
        "ImageNet16-120",
    )
    evaluated_indexes = set()
    target_full_dir = save_dir / target_dir
    target_less_dir = save_dir / "{:}-LESS".format(target_dir)
    arch_indexes = subdir2archs[target_full_dir]
    num_seeds = defaultdict(lambda: 0)
    end_time = time.time()
    arch_time = AverageMeter()
    for idx, arch_index in enumerate(arch_indexes):
        checkpoints = list(
            target_full_dir.glob("arch-{:}-seed-*.pth".format(arch_index))
        )
        ckps_less = list(target_less_dir.glob("arch-{:}-seed-*.pth".format(arch_index)))
        # create the arch info for each architecture
        try:
            arch_info_full = account_one_arch(
                arch_index,
                meta_archs[int(arch_index)],
                checkpoints,
                datasets,
                dataloader_dict,
            )
            arch_info_less = account_one_arch(
                arch_index,
                meta_archs[int(arch_index)],
                ckps_less,
                datasets,
                dataloader_dict,
            )
            num_seeds[len(checkpoints)] += 1
        except:
            print("Loading {:} failed, : {:}".format(arch_index, checkpoints))
            continue
        assert (
            int(arch_index) not in evaluated_indexes
        ), "conflict arch-index : {:}".format(arch_index)
        assert (
            0 <= int(arch_index) < len(meta_archs)
        ), "invalid arch-index {:} (not found in meta_archs)".format(arch_index)
        arch_info = {"full": arch_info_full, "less": arch_info_less}
        evaluated_indexes.add(int(arch_index))
        arch2infos[int(arch_index)] = arch_info
        # to correct the latency and training_time info.
        arch_info_full, arch_info_less = correct_time_related_info(
            int(arch_index), arch_info_full, arch_info_less
        )
        to_save_data = OrderedDict(
            full=arch_info_full.state_dict(), less=arch_info_less.state_dict()
        )
        torch.save(to_save_data, to_save_allarc / "{:}-FULL.pth".format(arch_index))
        arch_info["full"].clear_params()
        arch_info["less"].clear_params()
        torch.save(to_save_data, to_save_allarc / "{:}-SIMPLE.pth".format(arch_index))
        # measure elapsed time
        arch_time.update(time.time() - end_time)
        end_time = time.time()
        need_time = "{:}".format(
            convert_secs2time(arch_time.avg * (len(arch_indexes) - idx - 1), True)
        )
        print(
            "{:} {:} [{:03d}/{:03d}] : {:} still need {:}".format(
                time_string(), target_dir, idx, len(arch_indexes), arch_index, need_time
            )
        )
    # measure time
    xstrs = [
        "{:}:{:03d}".format(key, num_seeds[key])
        for key in sorted(list(num_seeds.keys()))
    ]
    print("{:} {:} done : {:}".format(time_string(), target_dir, xstrs))
    final_infos = {
        "meta_archs": meta_archs,
        "total_archs": meta_num_archs,
        "basestr": basestr,
        "arch2infos": arch2infos,
        "evaluated_indexes": evaluated_indexes,
    }
    save_file_name = to_save_simply / "{:}.pth".format(target_dir)
    torch.save(final_infos, save_file_name)
    print(
        "Save {:} / {:} architecture results into {:}.".format(
            len(evaluated_indexes), meta_num_archs, save_file_name
        )
    )


def merge_all(save_dir, meta_file, basestr):
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
    for index, sub_dir in enumerate(sub_model_dirs):
        arch_info_files = sorted(list(sub_dir.glob("arch-*-seed-*.pth")))
        print(
            "The {:02d}/{:02d}-th directory : {:} : {:} runs.".format(
                index, len(sub_model_dirs), sub_dir, len(arch_info_files)
            )
        )

    arch2infos, evaluated_indexes = dict(), set()
    for IDX, sub_dir in enumerate(sub_model_dirs):
        ckp_path = sub_dir.parent / "simplifies" / "{:}.pth".format(sub_dir.name)
        if ckp_path.exists():
            sub_ckps = torch.load(ckp_path, map_location="cpu")
            assert (
                sub_ckps["total_archs"] == meta_num_archs
                and sub_ckps["basestr"] == basestr
            )
            xarch2infos = sub_ckps["arch2infos"]
            xevalindexs = sub_ckps["evaluated_indexes"]
            for eval_index in xevalindexs:
                assert (
                    eval_index not in evaluated_indexes and eval_index not in arch2infos
                )
                # arch2infos[eval_index] = xarch2infos[eval_index].state_dict()
                arch2infos[eval_index] = {
                    "full": xarch2infos[eval_index]["full"].state_dict(),
                    "less": xarch2infos[eval_index]["less"].state_dict(),
                }
                evaluated_indexes.add(eval_index)
            print(
                "{:} [{:03d}/{:03d}] merge data from {:} with {:} models.".format(
                    time_string(), IDX, len(sub_model_dirs), ckp_path, len(xevalindexs)
                )
            )
        else:
            raise ValueError("Can not find {:}".format(ckp_path))
            # print ('{:} [{:03d}/{:03d}] can not find {:}, skip.'.format(time_string(), IDX, len(subdir2archs), ckp_path))

    evaluated_indexes = sorted(list(evaluated_indexes))
    print(
        "Finally, there are {:} architectures that have been trained and evaluated.".format(
            len(evaluated_indexes)
        )
    )

    to_save_simply = save_dir / "simplifies"
    if not to_save_simply.exists():
        to_save_simply.mkdir(parents=True, exist_ok=True)
    final_infos = {
        "meta_archs": meta_archs,
        "total_archs": meta_num_archs,
        "arch2infos": arch2infos,
        "evaluated_indexes": evaluated_indexes,
    }
    save_file_name = to_save_simply / "{:}-final-infos.pth".format(basestr)
    torch.save(final_infos, save_file_name)
    print(
        "Save {:} / {:} architecture results into {:}.".format(
            len(evaluated_indexes), meta_num_archs, save_file_name
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NAS-BENCH-201",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cal", "merge"],
        help="The running mode for this script.",
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="./output/NAS-BENCH-201-4",
        help="The base-name of folder to save checkpoints and log.",
    )
    parser.add_argument("--target_dir", type=str, help="The target directory.")
    parser.add_argument(
        "--max_node", type=int, default=4, help="The maximum node in a cell."
    )
    parser.add_argument(
        "--channel", type=int, default=16, help="The number of channels."
    )
    parser.add_argument(
        "--num_cells", type=int, default=5, help="The number of cells in one stage."
    )
    args = parser.parse_args()

    save_dir = Path(args.base_save_dir)
    meta_path = save_dir / "meta-node-{:}.pth".format(args.max_node)
    assert save_dir.exists(), "invalid save dir path : {:}".format(save_dir)
    assert meta_path.exists(), "invalid saved meta path : {:}".format(meta_path)
    print(
        "start the statistics of our nas-benchmark from {:} using {:}.".format(
            save_dir, args.target_dir
        )
    )
    basestr = "C{:}-N{:}".format(args.channel, args.num_cells)

    if args.mode == "cal":
        simplify(save_dir, meta_path, basestr, args.target_dir)
    elif args.mode == "merge":
        merge_all(save_dir, meta_path, basestr)
    else:
        raise ValueError("invalid mode : {:}".format(args.mode))
