##############################################################################
# NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07                          #
##############################################################################
# This file is used to train (all) architecture candidate in the topology    #
# search space in NATS-Bench (tss) with different hyper-parameters.          #
# When use mode=new, it will automatically detect whether the checkpoint of  #
# a trial exists, if so, it will skip this trial. When use mode=cover, it    #
# will ignore the (possible) existing checkpoint, run each trial, and save.  #
##############################################################################
# Please use the script of scripts/NATS-Bench/train-topology.sh to run.      #
# bash scripts/NATS-Bench/train-topology.sh 00000-15624 12 777               #
# bash scripts/NATS-Bench/train-topology.sh 00000-15624 200 '777 888 999'    #
#                                                                            #
################                                                             #
# [Deprecated Function: Generate the meta information]                       #
# python ./exps/NATS-Bench/main-tss.py --mode meta                           #
##############################################################################
import os, sys, time, torch, random, argparse
from typing import List, Text, Dict, Any
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

from xautodl.config_utils import dict2config, load_config
from xautodl.procedures import bench_evaluate_for_seed
from xautodl.procedures import get_machine_info
from xautodl.datasets import get_datasets
from xautodl.log_utils import Logger, AverageMeter, time_string, convert_secs2time
from xautodl.models import CellStructure, CellArchitectures, get_search_spaces
from xautodl.utils import split_str2indexes


def evaluate_all_datasets(
    arch: Text,
    datasets: List[Text],
    xpaths: List[Text],
    splits: List[Text],
    config_path: Text,
    seed: int,
    raw_arch_config,
    workers,
    logger,
):
    machine_info, raw_arch_config = get_machine_info(), deepcopy(raw_arch_config)
    all_infos = {"info": machine_info}
    all_dataset_keys = []
    # look all the datasets
    for dataset, xpath, split in zip(datasets, xpaths, splits):
        # train valid data
        train_data, valid_data, xshape, class_num = get_datasets(dataset, xpath, -1)
        # load the configuration
        if dataset == "cifar10" or dataset == "cifar100":
            split_info = load_config(
                "configs/nas-benchmark/cifar-split.txt", None, None
            )
        elif dataset.startswith("ImageNet16"):
            split_info = load_config(
                "configs/nas-benchmark/{:}-split.txt".format(dataset), None, None
            )
        else:
            raise ValueError("invalid dataset : {:}".format(dataset))
        config = load_config(
            config_path, dict(class_num=class_num, xshape=xshape), logger
        )
        # check whether use splited validation set
        if bool(split):
            assert dataset == "cifar10"
            ValLoaders = {
                "ori-test": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                )
            }
            assert len(train_data) == len(split_info.train) + len(
                split_info.valid
            ), "invalid length : {:} vs {:} + {:}".format(
                len(train_data), len(split_info.train), len(split_info.valid)
            )
            train_data_v2 = deepcopy(train_data)
            train_data_v2.transform = valid_data.transform
            valid_data = train_data_v2
            # data loader
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=config.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.train),
                num_workers=workers,
                pin_memory=True,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_data,
                batch_size=config.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.valid),
                num_workers=workers,
                pin_memory=True,
            )
            ValLoaders["x-valid"] = valid_loader
        else:
            # data loader
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=True,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_data,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
            )
            if dataset == "cifar10":
                ValLoaders = {"ori-test": valid_loader}
            elif dataset == "cifar100":
                cifar100_splits = load_config(
                    "configs/nas-benchmark/cifar100-test-split.txt", None, None
                )
                ValLoaders = {
                    "ori-test": valid_loader,
                    "x-valid": torch.utils.data.DataLoader(
                        valid_data,
                        batch_size=config.batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            cifar100_splits.xvalid
                        ),
                        num_workers=workers,
                        pin_memory=True,
                    ),
                    "x-test": torch.utils.data.DataLoader(
                        valid_data,
                        batch_size=config.batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            cifar100_splits.xtest
                        ),
                        num_workers=workers,
                        pin_memory=True,
                    ),
                }
            elif dataset == "ImageNet16-120":
                imagenet16_splits = load_config(
                    "configs/nas-benchmark/imagenet-16-120-test-split.txt", None, None
                )
                ValLoaders = {
                    "ori-test": valid_loader,
                    "x-valid": torch.utils.data.DataLoader(
                        valid_data,
                        batch_size=config.batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            imagenet16_splits.xvalid
                        ),
                        num_workers=workers,
                        pin_memory=True,
                    ),
                    "x-test": torch.utils.data.DataLoader(
                        valid_data,
                        batch_size=config.batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            imagenet16_splits.xtest
                        ),
                        num_workers=workers,
                        pin_memory=True,
                    ),
                }
            else:
                raise ValueError("invalid dataset : {:}".format(dataset))

        dataset_key = "{:}".format(dataset)
        if bool(split):
            dataset_key = dataset_key + "-valid"
        logger.log(
            "Evaluate ||||||| {:10s} ||||||| Train-Num={:}, Valid-Num={:}, Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
                dataset_key,
                len(train_data),
                len(valid_data),
                len(train_loader),
                len(valid_loader),
                config.batch_size,
            )
        )
        logger.log(
            "Evaluate ||||||| {:10s} ||||||| Config={:}".format(dataset_key, config)
        )
        for key, value in ValLoaders.items():
            logger.log(
                "Evaluate ---->>>> {:10s} with {:} batchs".format(key, len(value))
            )
        arch_config = dict2config(
            dict(
                name="infer.tiny",
                C=raw_arch_config["channel"],
                N=raw_arch_config["num_cells"],
                genotype=arch,
                num_classes=config.class_num,
            ),
            None,
        )
        results = bench_evaluate_for_seed(
            arch_config, config, train_loader, ValLoaders, seed, logger
        )
        all_infos[dataset_key] = results
        all_dataset_keys.append(dataset_key)
    all_infos["all_dataset_keys"] = all_dataset_keys
    return all_infos


def main(
    save_dir: Path,
    workers: int,
    datasets: List[Text],
    xpaths: List[Text],
    splits: List[int],
    seeds: List[int],
    nets: List[str],
    opt_config: Dict[Text, Any],
    to_evaluate_indexes: tuple,
    cover_mode: bool,
    arch_config: Dict[Text, Any],
):

    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(log_dir), os.getpid(), False)

    logger.log("xargs : seeds      = {:}".format(seeds))
    logger.log("xargs : cover_mode = {:}".format(cover_mode))
    logger.log("-" * 100)
    logger.log(
        "Start evaluating range =: {:06d} - {:06d}".format(
            min(to_evaluate_indexes), max(to_evaluate_indexes)
        )
        + "({:} in total) / {:06d} with cover-mode={:}".format(
            len(to_evaluate_indexes), len(nets), cover_mode
        )
    )
    for i, (dataset, xpath, split) in enumerate(zip(datasets, xpaths, splits)):
        logger.log(
            "--->>> Evaluate {:}/{:} : dataset={:9s}, path={:}, split={:}".format(
                i, len(datasets), dataset, xpath, split
            )
        )
    logger.log("--->>> optimization config : {:}".format(opt_config))

    start_time, epoch_time = time.time(), AverageMeter()
    for i, index in enumerate(to_evaluate_indexes):
        arch = nets[index]
        logger.log(
            "\n{:} evaluate {:06d}/{:06d} ({:06d}/{:06d})-th arch [seeds={:}] {:}".format(
                time_string(),
                i,
                len(to_evaluate_indexes),
                index,
                len(nets),
                seeds,
                "-" * 15,
            )
        )
        logger.log("{:} {:} {:}".format("-" * 15, arch, "-" * 15))

        # test this arch on different datasets with different seeds
        has_continue = False
        for seed in seeds:
            to_save_name = save_dir / "arch-{:06d}-seed-{:04d}.pth".format(index, seed)
            if to_save_name.exists():
                if cover_mode:
                    logger.log(
                        "Find existing file : {:}, remove it before evaluation".format(
                            to_save_name
                        )
                    )
                    os.remove(str(to_save_name))
                else:
                    logger.log(
                        "Find existing file : {:}, skip this evaluation".format(
                            to_save_name
                        )
                    )
                    has_continue = True
                    continue
            results = evaluate_all_datasets(
                CellStructure.str2structure(arch),
                datasets,
                xpaths,
                splits,
                opt_config,
                seed,
                arch_config,
                workers,
                logger,
            )
            torch.save(results, to_save_name)
            logger.log(
                "\n{:} evaluate {:06d}/{:06d} ({:06d}/{:06d})-th arch [seeds={:}] ===>>> {:}".format(
                    time_string(),
                    i,
                    len(to_evaluate_indexes),
                    index,
                    len(nets),
                    seeds,
                    to_save_name,
                )
            )
        # measure elapsed time
        if not has_continue:
            epoch_time.update(time.time() - start_time)
        start_time = time.time()
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.avg * (len(to_evaluate_indexes) - i - 1), True)
        )
        logger.log(
            "This arch costs : {:}".format(convert_secs2time(epoch_time.val, True))
        )
        logger.log("{:}".format("*" * 100))
        logger.log(
            "{:}   {:74s}   {:}".format(
                "*" * 10,
                "{:06d}/{:06d} ({:06d}/{:06d})-th done, left {:}".format(
                    i, len(to_evaluate_indexes), index, len(nets), need_time
                ),
                "*" * 10,
            )
        )
        logger.log("{:}".format("*" * 100))

    logger.close()


def train_single_model(
    save_dir, workers, datasets, xpaths, splits, use_less, seeds, model_str, arch_config
):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.set_num_threads(workers)

    save_dir = (
        Path(save_dir)
        / "specifics"
        / "{:}-{:}-{:}-{:}".format(
            "LESS" if use_less else "FULL",
            model_str,
            arch_config["channel"],
            arch_config["num_cells"],
        )
    )
    logger = Logger(str(save_dir), 0, False)
    if model_str in CellArchitectures:
        arch = CellArchitectures[model_str]
        logger.log(
            "The model string is found in pre-defined architecture dict : {:}".format(
                model_str
            )
        )
    else:
        try:
            arch = CellStructure.str2structure(model_str)
        except:
            raise ValueError(
                "Invalid model string : {:}. It can not be found or parsed.".format(
                    model_str
                )
            )
    assert arch.check_valid_op(
        get_search_spaces("cell", "full")
    ), "{:} has the invalid op.".format(arch)
    logger.log("Start train-evaluate {:}".format(arch.tostr()))
    logger.log("arch_config : {:}".format(arch_config))

    start_time, seed_time = time.time(), AverageMeter()
    for _is, seed in enumerate(seeds):
        logger.log(
            "\nThe {:02d}/{:02d}-th seed is {:} ----------------------<.>----------------------".format(
                _is, len(seeds), seed
            )
        )
        to_save_name = save_dir / "seed-{:04d}.pth".format(seed)
        if to_save_name.exists():
            logger.log(
                "Find the existing file {:}, directly load!".format(to_save_name)
            )
            checkpoint = torch.load(to_save_name)
        else:
            logger.log(
                "Does not find the existing file {:}, train and evaluate!".format(
                    to_save_name
                )
            )
            checkpoint = evaluate_all_datasets(
                arch,
                datasets,
                xpaths,
                splits,
                use_less,
                seed,
                arch_config,
                workers,
                logger,
            )
            torch.save(checkpoint, to_save_name)
        # log information
        logger.log("{:}".format(checkpoint["info"]))
        all_dataset_keys = checkpoint["all_dataset_keys"]
        for dataset_key in all_dataset_keys:
            logger.log(
                "\n{:} dataset : {:} {:}".format("-" * 15, dataset_key, "-" * 15)
            )
            dataset_info = checkpoint[dataset_key]
            # logger.log('Network ==>\n{:}'.format( dataset_info['net_string'] ))
            logger.log(
                "Flops = {:} MB, Params = {:} MB".format(
                    dataset_info["flop"], dataset_info["param"]
                )
            )
            logger.log("config : {:}".format(dataset_info["config"]))
            logger.log(
                "Training State (finish) = {:}".format(dataset_info["finish-train"])
            )
            last_epoch = dataset_info["total_epoch"] - 1
            train_acc1es, train_acc5es = (
                dataset_info["train_acc1es"],
                dataset_info["train_acc5es"],
            )
            valid_acc1es, valid_acc5es = (
                dataset_info["valid_acc1es"],
                dataset_info["valid_acc5es"],
            )
            logger.log(
                "Last Info : Train = Acc@1 {:.2f}% Acc@5 {:.2f}% Error@1 {:.2f}%, Test = Acc@1 {:.2f}% Acc@5 {:.2f}% Error@1 {:.2f}%".format(
                    train_acc1es[last_epoch],
                    train_acc5es[last_epoch],
                    100 - train_acc1es[last_epoch],
                    valid_acc1es[last_epoch],
                    valid_acc5es[last_epoch],
                    100 - valid_acc1es[last_epoch],
                )
            )
        # measure elapsed time
        seed_time.update(time.time() - start_time)
        start_time = time.time()
        need_time = "Time Left: {:}".format(
            convert_secs2time(seed_time.avg * (len(seeds) - _is - 1), True)
        )
        logger.log(
            "\n<<<***>>> The {:02d}/{:02d}-th seed is {:} <finish> other procedures need {:}".format(
                _is, len(seeds), seed, need_time
            )
        )
    logger.close()


def generate_meta_info(save_dir, max_node, divide=40):
    aa_nas_bench_ss = get_search_spaces("cell", "nas-bench-201")
    archs = CellStructure.gen_all(aa_nas_bench_ss, max_node, False)
    print(
        "There are {:} archs vs {:}.".format(
            len(archs), len(aa_nas_bench_ss) ** ((max_node - 1) * max_node / 2)
        )
    )

    random.seed(88)  # please do not change this line for reproducibility
    random.shuffle(archs)
    # to test fixed-random shuffle
    # print ('arch [0] : {:}\n---->>>>   {:}'.format( archs[0], archs[0].tostr() ))
    # print ('arch [9] : {:}\n---->>>>   {:}'.format( archs[9], archs[9].tostr() ))
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
    total_arch = len(archs)

    num = 50000
    indexes_5W = list(range(num))
    random.seed(1021)
    random.shuffle(indexes_5W)
    train_split = sorted(list(set(indexes_5W[: num // 2])))
    valid_split = sorted(list(set(indexes_5W[num // 2 :])))
    assert len(train_split) + len(valid_split) == num
    assert (
        train_split[0] == 0
        and train_split[10] == 26
        and train_split[111] == 203
        and valid_split[0] == 1
        and valid_split[10] == 18
        and valid_split[111] == 242
    ), "{:} {:} {:} - {:} {:} {:}".format(
        train_split[0],
        train_split[10],
        train_split[111],
        valid_split[0],
        valid_split[10],
        valid_split[111],
    )
    splits = {num: {"train": train_split, "valid": valid_split}}

    info = {
        "archs": [x.tostr() for x in archs],
        "total": total_arch,
        "max_node": max_node,
        "splits": splits,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = save_dir / "meta-node-{:}.pth".format(max_node)
    assert not save_name.exists(), "{:} already exist".format(save_name)
    torch.save(info, save_name)
    print("save the meta file into {:}".format(save_name))


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


def filter_indexes(xlist, mode, save_dir, seeds):
    all_indexes = []
    for index in xlist:
        if mode == "cover":
            all_indexes.append(index)
        else:
            for seed in seeds:
                temp_path = save_dir / "arch-{:06d}-seed-{:04d}.pth".format(index, seed)
                if not temp_path.exists():
                    all_indexes.append(index)
                    break
    print(
        "{:} [FILTER-INDEXES] : there are {:}/{:} architectures in total".format(
            time_string(), len(all_indexes), len(xlist)
        )
    )
    return all_indexes


if __name__ == "__main__":
    # mode_choices = ['meta', 'new', 'cover'] + ['specific-{:}'.format(_) for _ in CellArchitectures.keys()]
    parser = argparse.ArgumentParser(
        description="NATS-Bench (topology search space)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True, help="The script mode.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/NATS-Bench-topology",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--max_node",
        type=int,
        default=4,
        help="The maximum node in a cell (please do not change it).",
    )
    # use for train the model
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--srange", type=str, required=True, help="The range of models to be evaluated"
    )
    parser.add_argument("--datasets", type=str, nargs="+", help="The applied datasets.")
    parser.add_argument(
        "--xpaths", type=str, nargs="+", help="The root path for this dataset."
    )
    parser.add_argument(
        "--splits", type=int, nargs="+", help="The root path for this dataset."
    )
    parser.add_argument(
        "--hyper",
        type=str,
        default="12",
        choices=["01", "12", "200"],
        help="The tag for hyper-parameters.",
    )

    parser.add_argument(
        "--seeds", type=int, nargs="+", help="The range of models to be evaluated"
    )
    parser.add_argument(
        "--channel", type=int, default=16, help="The number of channels."
    )
    parser.add_argument(
        "--num_cells", type=int, default=5, help="The number of cells in one stage."
    )
    parser.add_argument("--check_N", type=int, default=15625, help="For safety.")
    args = parser.parse_args()

    assert args.mode in ["meta", "new", "cover"] or args.mode.startswith(
        "specific-"
    ), "invalid mode : {:}".format(args.mode)

    if args.mode == "meta":
        generate_meta_info(args.save_dir, args.max_node)
    elif args.mode.startswith("specific"):
        assert len(args.mode.split("-")) == 2, "invalid mode : {:}".format(args.mode)
        model_str = args.mode.split("-")[1]
        train_single_model(
            args.save_dir,
            args.workers,
            args.datasets,
            args.xpaths,
            args.splits,
            args.use_less > 0,
            tuple(args.seeds),
            model_str,
            {"channel": args.channel, "num_cells": args.num_cells},
        )
    else:
        nets = traverse_net(args.max_node)
        if len(nets) != args.check_N:
            raise ValueError(
                "Pre-num-check failed : {:} vs {:}".format(len(nets), args.check_N)
            )
        opt_config = "./configs/nas-benchmark/hyper-opts/{:}E.config".format(args.hyper)
        if not os.path.isfile(opt_config):
            raise ValueError("{:} is not a file.".format(opt_config))
        save_dir = Path(args.save_dir) / "raw-data-{:}".format(args.hyper)
        save_dir.mkdir(parents=True, exist_ok=True)
        to_evaluate_indexes = split_str2indexes(args.srange, args.check_N, 5)
        if not len(args.seeds):
            raise ValueError("invalid length of seeds args: {:}".format(args.seeds))
        if not (len(args.datasets) == len(args.xpaths) == len(args.splits)):
            raise ValueError(
                "invalid infos : {:} vs {:} vs {:}".format(
                    len(args.datasets), len(args.xpaths), len(args.splits)
                )
            )
        if args.workers < 0:
            raise ValueError("invalid number of workers : {:}".format(args.workers))

        target_indexes = filter_indexes(
            to_evaluate_indexes, args.mode, save_dir, args.seeds
        )

        assert torch.cuda.is_available(), "CUDA is not available."
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        # torch.set_num_threads(args.workers if args.workers > 0 else 1)

        main(
            save_dir,
            args.workers,
            args.datasets,
            args.xpaths,
            args.splits,
            tuple(args.seeds),
            nets,
            opt_config,
            target_indexes,
            args.mode == "cover",
            {
                "name": "infer.tiny",
                "channel": args.channel,
                "num_cells": args.num_cells,
            },
        )
