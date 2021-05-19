##############################################################################
# NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07                          #
##############################################################################
# This file is used to train (all) architecture candidate in the size search #
# space in NATS-Bench (sss) with different hyper-parameters.                 #
# When use mode=new, it will automatically detect whether the checkpoint of  #
# a trial exists, if so, it will skip this trial. When use mode=cover, it    #
# will ignore the (possible) existing checkpoint, run each trial, and save.  #
# (NOTE): the topology for all candidates in sss is fixed as:                ######################
# |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2| #
###################################################################################################
# Please use the script of scripts/NATS-Bench/train-shapes.sh to run.        #
##############################################################################
import os, sys, time, torch, argparse
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
from xautodl.utils import split_str2indexes


def evaluate_all_datasets(
    channels: Text,
    datasets: List[Text],
    xpaths: List[Text],
    splits: List[Text],
    config_path: Text,
    seed: int,
    workers: int,
    logger,
):
    machine_info = get_machine_info()
    all_infos = {"info": machine_info}
    all_dataset_keys = []
    # look all the dataset
    for dataset, xpath, split in zip(datasets, xpaths, splits):
        # the train and valid data
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
        # check whether use the splitted validation set
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
        # arch-index= 9930, arch=|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
        # this genotype is the architecture with the highest accuracy on CIFAR-100 validation set
        genotype = "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|"
        arch_config = dict2config(
            dict(
                name="infer.shape.tiny",
                channels=channels,
                genotype=genotype,
                num_classes=class_num,
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
        channelstr = nets[index]
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
        logger.log("{:} {:} {:}".format("-" * 15, channelstr, "-" * 15))

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
                channelstr, datasets, xpaths, splits, opt_config, seed, workers, logger
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


def traverse_net(candidates: List[int], N: int):
    nets = [""]
    for i in range(N):
        new_nets = []
        for net in nets:
            for C in candidates:
                new_nets.append(str(C) if net == "" else "{:}:{:}".format(net, C))
        nets = new_nets
    return nets


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

    SLURM_PROCID, SLURM_NTASKS = "SLURM_PROCID", "SLURM_NTASKS"
    if SLURM_PROCID in os.environ and SLURM_NTASKS in os.environ:  # run on the slurm
        proc_id, ntasks = int(os.environ[SLURM_PROCID]), int(os.environ[SLURM_NTASKS])
        assert 0 <= proc_id < ntasks, "invalid proc_id {:} vs ntasks {:}".format(
            proc_id, ntasks
        )
        scales = [int(float(i) / ntasks * len(all_indexes)) for i in range(ntasks)] + [
            len(all_indexes)
        ]
        per_job = []
        for i in range(ntasks):
            xs, xe = min(max(scales[i], 0), len(all_indexes) - 1), min(
                max(scales[i + 1] - 1, 0), len(all_indexes) - 1
            )
            per_job.append((xs, xe))
        for i, srange in enumerate(per_job):
            print("  -->> {:2d}/{:02d} : {:}".format(i, ntasks, srange))
        current_range = per_job[proc_id]
        all_indexes = [
            all_indexes[i] for i in range(current_range[0], current_range[1] + 1)
        ]
        # set the device id
        device = proc_id % torch.cuda.device_count()
        torch.cuda.set_device(device)
        print("  set the device id = {:}".format(device))
    print(
        "{:} [FILTER-INDEXES] : after filtering there are {:} architectures in total".format(
            time_string(), len(all_indexes)
        )
    )
    return all_indexes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NATS-Bench (size search space)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["new", "cover"],
        help="The script mode.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/NATS-Bench-size",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--candidateC",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32, 40, 48, 56, 64],
        help=".",
    )
    parser.add_argument(
        "--num_layers", type=int, default=5, help="The number of layers in a network."
    )
    parser.add_argument("--check_N", type=int, default=32768, help="For safety.")
    # use for train the model
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="The number of data loading workers (default: 2)",
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
        choices=["01", "12", "90"],
        help="The tag for hyper-parameters.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="The range of models to be evaluated"
    )
    args = parser.parse_args()

    nets = traverse_net(args.candidateC, args.num_layers)
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
    if args.workers <= 0:
        raise ValueError("invalid number of workers : {:}".format(args.workers))

    target_indexes = filter_indexes(
        to_evaluate_indexes, args.mode, save_dir, args.seeds
    )

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    # torch.set_num_threads(args.workers)

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
    )
