##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##############################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, SearchDataset
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_search_spaces

from nas_201_api import NASBench201API as API
from R_EA import train_and_eval, random_architecture_func


def main(xargs, nas_bench):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    if xargs.dataset == "cifar10":
        dataname = "cifar10-valid"
    else:
        dataname = xargs.dataset
    if xargs.data_path is not None:
        train_data, valid_data, xshape, class_num = get_datasets(
            xargs.dataset, xargs.data_path, -1
        )
        split_Fpath = "configs/nas-benchmark/cifar-split.txt"
        cifar_split = load_config(split_Fpath, None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid
        logger.log("Load split file from {:}".format(split_Fpath))
        config_path = "configs/nas-benchmark/algos/R-EA.config"
        config = load_config(
            config_path, {"class_num": class_num, "xshape": xshape}, logger
        )
        # To split data
        train_data_v2 = deepcopy(train_data)
        train_data_v2.transform = valid_data.transform
        valid_data = train_data_v2
        search_data = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            num_workers=xargs.workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
            num_workers=xargs.workers,
            pin_memory=True,
        )
        logger.log(
            "||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
                xargs.dataset, len(train_loader), len(valid_loader), config.batch_size
            )
        )
        logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))
        extra_info = {
            "config": config,
            "train_loader": train_loader,
            "valid_loader": valid_loader,
        }
    else:
        config_path = "configs/nas-benchmark/algos/R-EA.config"
        config = load_config(config_path, None, logger)
        logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))
        extra_info = {"config": config, "train_loader": None, "valid_loader": None}
    search_space = get_search_spaces("cell", xargs.search_space_name)
    random_arch = random_architecture_func(xargs.max_nodes, search_space)
    # x =random_arch() ; y = mutate_arch(x)
    x_start_time = time.time()
    logger.log("{:} use nas_bench : {:}".format(time_string(), nas_bench))
    best_arch, best_acc, total_time_cost, history = None, -1, 0, []
    # for idx in range(xargs.random_num):
    while total_time_cost < xargs.time_budget:
        arch = random_arch()
        accuracy, cost_time = train_and_eval(arch, nas_bench, extra_info, dataname)
        if total_time_cost + cost_time > xargs.time_budget:
            break
        else:
            total_time_cost += cost_time
        history.append(arch)
        if best_arch is None or best_acc < accuracy:
            best_acc, best_arch = accuracy, arch
        logger.log(
            "[{:03d}] : {:} : accuracy = {:.2f}%".format(len(history), arch, accuracy)
        )
    logger.log(
        "{:} best arch is {:}, accuracy = {:.2f}%, visit {:} archs with {:.1f} s (real-cost = {:.3f} s).".format(
            time_string(),
            best_arch,
            best_acc,
            len(history),
            total_time_cost,
            time.time() - x_start_time,
        )
    )

    info = nas_bench.query_by_arch(best_arch, "200")
    if info is None:
        logger.log("Did not find this architecture : {:}.".format(best_arch))
    else:
        logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()
    return logger.log_dir, nas_bench.query_index_by_arch(best_arch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random NAS")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # channels and number-of-cells
    parser.add_argument("--search_space_name", type=str, help="The search space name.")
    parser.add_argument("--max_nodes", type=int, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, help="The number of channels.")
    parser.add_argument(
        "--num_cells", type=int, help="The number of cells in one stage."
    )
    # parser.add_argument('--random_num',         type=int,   help='The number of random selected architectures.')
    parser.add_argument(
        "--time_budget",
        type=int,
        help="The total time cost budge for searching (in seconds).",
    )
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log."
    )
    parser.add_argument(
        "--arch_nas_dataset",
        type=str,
        help="The path to load the architecture dataset (tiny-nas-benchmark).",
    )
    parser.add_argument("--print_freq", type=int, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", type=int, help="manual seed")
    args = parser.parse_args()
    # if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    if args.arch_nas_dataset is None or not os.path.isfile(args.arch_nas_dataset):
        nas_bench = None
    else:
        print(
            "{:} build NAS-Benchmark-API from {:}".format(
                time_string(), args.arch_nas_dataset
            )
        )
        nas_bench = API(args.arch_nas_dataset)
    if args.rand_seed < 0:
        save_dir, all_indexes, num = None, [], 500
        for i in range(num):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, num))
            args.rand_seed = random.randint(1, 100000)
            save_dir, index = main(args, nas_bench)
            all_indexes.append(index)
        torch.save(all_indexes, save_dir / "results.pth")
    else:
        main(args, nas_bench)
